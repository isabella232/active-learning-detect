## Management CLI

Data management CLI to interact with data manager endpoints.

This CLI attempts to be as simple as possible to allow uers to initialize a dataset, download a dataset, and upload a dataset to an Azure Storage blob. It presumes you have a functioning management endpoint and database.

### Configuration

Create an INI file, and store it anywhere. Copy the path to it, and add it to your environment variables as `ALCONFIG`

Example: `export ALCONFIG=/path/to/config.ini`

The INI file should contain the following sections and keys to operate properly

```
[FUNCTIONS]
FUNCTIONS_KEY=
FUNCTIONS_URL=https://mytagmanagement.azurewebsites.net/

[STORAGE]
STORAGE_ACCOUNT=
STORAGE_KEY=
STORAGE_CONTAINER=vott

[TAGGING]
TAGGING_LOCATION=~/taggingdata
TAGGING_USER=bhargav
TAGGING_IMAGE_DIR=${TAGGING:TAGGING_LOCATION}/AllImages
```

`FUNCTIONS_KEY` is the Azure Functions Key that allows your CLI to authenticate with the management function
`FUNCTIONS_URL` is the URL of the Function deployed to Azure

`STORAGE_ACCOUNT` is the name of the Azure Storage Account used to upload images
`STORAGE_KEY` is the secret key of the Azure Storage Account
`STORAGE_CONTAINER` is the name of the container where the CLI deposits your image files.

`TAGGING_LOCATION` is the location on the user's machine where media will be downloaded
`TAGGING_USER` is your username.
`TAGGING_IMAGE_DIR` is the location where images will be downloaded, usually /AllImages folder inside tagging_location. 

`TAGGING_IMAGE_DIR` is the location where all images will be downloaded to for training
`TAGGED_OUTPUT` is the location of the CSV file that will have human labelled data

### Commands

#### Initialize a dataset/Onboard an existing dataset.

Usage: `python3 -m cli.cli onboard -f /path/to/images/`

Assuming your directory `/path/to/images` is a flat directory of images, you can use this CLI invocation to upload your images to a temporary storage container.

The onboarding function is then invoked, processing your images into the database, making them available for downloading.

#### Download

Usage: `python3 -m cli.cli download -n 50`

Downloads 50 images to the location identified by `TAGGING_LOCATION` in your config.
There is an upper bound of 100 images that can be downloaded at present.

Also generated is a VoTT json file containing any existing tags and labels.

#### Upload tags

Usage: `python3 -m cli.cli upload`

Uploads the VoTT json file to be processed into the database. Will also delete the image directory
identified at `TAGGING_LOCATION`, so the next `download` cycle will commence without issue.
