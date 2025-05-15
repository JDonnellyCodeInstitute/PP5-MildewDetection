from pathlib import Path
import requests

def download_model_from_google_drive(file_id: str, destination: Path):
    """
    Download a file from Google Drive into `destination`.
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url, stream=True)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def get_or_download_model(file_id: str, dest_path: Path):
    """
    Ensure the Keras .h5 model exists at dest_path; if not, fetch from Drive.
    Returns the loaded model.
    """
    from tensorflow.keras.models import load_model

    if not dest_path.exists():
        print(f"Downloading model to {dest_path}…")
        download_model_from_google_drive(file_id, dest_path)
    return load_model(dest_path)
