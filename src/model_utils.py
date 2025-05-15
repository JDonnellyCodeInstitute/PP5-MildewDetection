from pathlib import Path
import gdown
from tensorflow.keras.models import load_model

DRIVE_URL = "https://drive.google.com/uc?id={file_id}"

def get_or_download_model(file_id: str, dest_path: Path):
    """
    Download the model via gdown if missing, then load it.
    """
    if not dest_path.exists():
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        url = DRIVE_URL.format(file_id=file_id)
        print(f"Downloading model from Drive via gdown: {url}")
        gdown.download(url, str(dest_path), quiet=False)
    return load_model(dest_path)