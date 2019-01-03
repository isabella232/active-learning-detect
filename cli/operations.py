import requests
import time
import shutil
import json
import copy
import pathlib
import os
from urlpath import URL
from azure.storage.blob import BlockBlobService, ContentSettings
from utils.blob_utils import BlobStorage
from utils.vott_parser import process_vott_json, create_starting_vott_json, build_id_to_VottImageTag, create_vott_json_from_image_labels
from functions.pipeline.shared.db_access import ImageLabel, ImageTag

DEFAULT_NUM_IMAGES = 40
LOWER_LIMIT = 0
UPPER_LIMIT = 100

azure_storage_client = None


class ImageLimitException(Exception):
    pass


def supported_file_type(file_name):
    if file_name.startswith('.'):
        return False

    file_suffix = pathlib.Path(file_name).suffix.lower()
    if file_suffix in ['.png', '.jpg', '.jpeg', '.gif']:
        return True
    else:
        return False


# Somewhat of a hack to remove the folder name passed on the
# command line from the file path to make blob names more sane.
# If one invokes the cli with the onboard folder option and passes
# along a path to a folder, every file path will contain the folder prefix.
# Example invocation: python3 -m cli onboard /my/full/path
# os.walk will return: /my/full/path/1.jpg, /my/full/path/2.jpg, etc.
def strip_path_prefix(folder_name, file_path):
    folder_name_str = str(folder_name)
    file_path_str = str(file_path)

    stripped_path = file_path_str.replace(folder_name_str, "")

    if stripped_path.startswith('/'):
        return pathlib.Path(stripped_path[1:])

    return pathlib.Path(stripped_path)


# TODO We should create the container if it does not exist
def onboard_folder(config, folder_name):
    blob_storage = BlobStorage.get_azure_storage_client(config)
    user_name = config.get("tagging_user")
    onboarding_files = []

    print("Walking file system")

    for (root, dirs, files) in os.walk(folder_name):
        # no files at this level.
        if len(files) == 0:
            continue

        for file_name in files:
            if not supported_file_type(file_name):
                continue

            relative_path = os.path.join(root, file_name)
            onboarding_files.append(relative_path)

    if len(onboarding_files) == 0:
        print(f'Could not find any valid files to upload')
        return

    for blob_path in onboarding_files:
        stripped_path = strip_path_prefix(folder_name, blob_path)
        print("Uploading " + str(blob_path))

        blob_metadata = {
            "userFilePath": blob_path,
            "uploadUser": config.get("tagging_user")
        }

        blob_storage.create_blob_from_path(
            config.get("storage_temp_container"),
            stripped_path,  # the name of the file in blob storage.
            pathlib.Path(blob_path),
            content_settings=ContentSettings(content_type='image/png'),
            metadata=blob_metadata
        )

    # Trigger queue based onboarding.
    onboard_container(
        config,
        config.get('storage_account'),
        config.get('storage_key'),
        config.get('storage_container')
    )


def onboard_container(config, account, key, container):
    print("onboarding from storage container")
    function_url = config.get('url') + '/api/onboardcontainer'
    user_name = config.get("tagging_user")

    print("Onboarding storage container " + container + " into dataset")

    query = {
        "userName": user_name
    }

    data = {
        "storageAccount": account,
        "storageAccountKey": key,
        "storageContainer": container
    }

    resp = requests.post(function_url, params=query, json=data)
    resp.raise_for_status()

    print("Set up container for onboarding. Onboarding may take some time.")


def _download_bounds(num_images):
    images_to_download = num_images

    if num_images is None:
        images_to_download = DEFAULT_NUM_IMAGES

    if images_to_download <= LOWER_LIMIT or images_to_download > UPPER_LIMIT:
        raise ImageLimitException()

    return images_to_download


def download(config, num_images, strategy=None):
    # TODO: better/more proper URI handling.
    functions_url = config.get("url") + "/api/images"
    user_name = config.get("tagging_user")
    images_to_download = _download_bounds(num_images)
    query = {
        "imageCount": images_to_download,
        "userName": user_name,
        "checkOut": "true"
    }

    response = requests.get(functions_url, params=query)
    response.raise_for_status()

    json_resp = response.json()
    images_json = json.loads(json_resp["images"])
    count = len(images_json)

    print("Received " + str(count) + " files.")

    if count == 0:
        print("No images could be retrieved with the current retrieval strategy!")
        return

    file_tree = pathlib.Path(os.path.expanduser(
        config.get("tagging_location"))
    )

    if file_tree.exists():
        print("Removing existing tag data directory: " + str(file_tree))

        shutil.rmtree(str(file_tree), ignore_errors=True)

    data_dir = pathlib.Path(file_tree / "data")
    data_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    checkedout_image_labels = [ImageLabel.fromJson(item) for item in images_json]
    vott_json, image_urls = create_vott_json_from_image_labels(checkedout_image_labels, json_resp["classification_list"])

    json_data = {'vott_json': vott_json,
                 'imageUrls': image_urls}

    local_images = download_images(config, data_dir, json_data)
    count = len(local_images)
    print("Successfully downloaded " + str(count) + " images.")
    for image_path in local_images:
        print(image_path)
    print("Ready to tag!")


def download_images(config, image_dir, json_resp):
    print("Downloading files to " + str(image_dir))

    # Write generated VoTT data from the function to a file.
    write_vott_data(image_dir, json_resp)

    urls = json_resp['imageUrls']
    downloaded_file_paths = []
    for url_path in urls:
        url = URL(url_path)

        # TODO: We will download an empty file if we get a permission error on the blob store URL
        # We should raise an exception. For now the blob store must be publically accessible
        response = requests.get(url)
        file_path = pathlib.Path(image_dir / url.name)

        with open(str(file_path), "wb") as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
            file.close()
        downloaded_file_paths.append(file_path)
    return downloaded_file_paths


def write_vott_data(image_dir, json_resp):
    # VOTT expects json file at same level as directory
    data_file = pathlib.Path(image_dir / "../data.json")
    vott_data = json_resp.get("vott_json", None)

    if not vott_data:
        return

    with open(str(data_file), "w") as file:
        vott_json_string = json.dumps(vott_data)
        file.writelines(vott_json_string)
        file.close()


def upload(config):
    functions_url = config.get("url") + "/api/labels"
    user_name = config.get("tagging_user")
    tagging_location = pathlib.Path(
        os.path.expanduser(config.get("tagging_location"))
    )

    print("Uploading VOTT json file...")
    vott_json = pathlib.Path(tagging_location / "data.json")

    with open(str(vott_json)) as json_file:
        json_data = json.load(json_file)
    process_json = process_vott_json(json_data)
    query = {
        "userName": user_name,
        "upload": "true"
    }

    response = requests.post(functions_url, json=process_json, params=query)
    response.raise_for_status()

    resp_json = response.json()
    print("Done!")