import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset(
    dataset="praveengovi/emotions-dataset-for-nlp",
    download_path="data"
):
    """
    Downloads and extracts a Kaggle dataset using Kaggle API.
    """

    # Create download directory
    os.makedirs(download_path, exist_ok=True)

    # Authenticate
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset: {dataset}")

    # Download dataset
    api.dataset_download_files(
        dataset,
        path=download_path,
        unzip=False
    )

    # Find downloaded zip file
    zip_files = [f for f in os.listdir(download_path) if f.endswith(".zip")]

    if not zip_files:
        raise FileNotFoundError("Dataset download failed. No zip file found.")

    zip_path = os.path.join(download_path, zip_files[0])

    print(f"Extracting {zip_path}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)

    print("Dataset ready.")

    return download_path

