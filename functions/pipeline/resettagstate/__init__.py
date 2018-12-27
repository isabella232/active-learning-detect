import datetime
import logging

import azure.functions as func
from ..shared.db_provider import get_postgres_provider
from ..shared.db_access import ImageTagDataAccess, ImageInfo


def main(mytimer: func.TimerRequest) -> None:
    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()

    try:
        data_access = ImageTagDataAccess(get_postgres_provider())
        stale_image_ids = data_access.reset_stale_checkedout_images()
        logging.info('Reset {} images to TAGGING_INCOMPLETE state'.format(len(stale_image_ids)))
    except Exception as e:
        logging.error('Error encounted while trying to reset stale image states. {}'.format(e))
        raise

    logging.info('Successfully ran reset function at %s', utc_timestamp)
