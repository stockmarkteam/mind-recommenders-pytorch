import os
import zipfile
from logging import getLogger

logger = getLogger(__name__)


def decompress_dataset(
    train_zip: str,
    valid_zip: str,
    dataset_type: str,
    output_dir: str,
) -> None:
    """
    Description:
        script for extract dataset.
        save path:
            train_data: {output_dir}/train
            valid_data : {output_dir}/valid
    """
    logger.info(f"--- extract {dataset_type} dataset ---")
    os.makedirs(output_dir, exist_ok=True)
    for data_category, zip_path in zip(["train", "valid"], [train_zip, valid_zip]):
        if zip_path is not None:
            logger.info(f"decompressing {data_category} dataset...")
            zipfile.ZipFile(zip_path).extractall(os.path.join(output_dir, data_category))
