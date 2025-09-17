from loguru import logger
from tqdm import tqdm
import typer
import requests
import zipfile
from colours_of_red.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def download_rapsberry_dataset(dataset: str) -> None:
    """
    Downloads the RaspberrySet: Dataset of Annotated Raspberry Images for Object Detection
    from Zenodo (http://zenodo.org/records/7014728).
    Saves the dataset to the specified output path with progress tracking.
    """
    url = "https://zenodo.org/records/7014728/files/RaspberrySet.zip"
    logger.info(f"Starting download of RaspberrySet from {url}")
    
    # Define input and output paths based on dataset name
    download_path = EXTERNAL_DATA_DIR / f"{dataset}.zip"   
    logger.info(f"Downloading to: {download_path}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        download_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(download_path, "wb") as f, tqdm(
            desc="Downloading RaspberrySet dataset",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        logger.success(f"RaspberrySet dataset downloaded successfully to {download_path}")

        
        extract_path = PROCESSED_DATA_DIR / dataset
        logger.info(f"Extracting to: {extract_path}")
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        
        logger.success(f"RaspberrySet dataset extracted successfully to {extract_path}")

    except requests.RequestException as e:
        logger.error(f"Error downloading RaspberrySet dataset: {e}")
        raise

@app.command()
def main(
    dataset: str = typer.Option("raspberryset", help="Name of the dataset to download (e.g., 'raspberryset')")
):
    """
    Entry point to download datasets, with paths defined relative to the dataset name.
    """
    logger.info(f"Processing dataset request: {dataset}")
    
    if dataset.lower() == "raspberryset":
        download_rapsberry_dataset(dataset)
    else:
        logger.warning(f"Dataset '{dataset}' is not supported yet.")
    
    logger.success("Dataset processing complete.")

if __name__ == "__main__":
    app()