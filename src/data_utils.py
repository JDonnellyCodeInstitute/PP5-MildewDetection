from pathlib import Path
import shutil, random, os

def clean_image_dataset(root_dir, extensions=None):
    """
    Remove any files in each subdirectory of root_dir whose suffix
    is not in the allowed extensions, and print a summary.
    """
    # Default to common image suffixes if none are provided
    allowed = {'.png', '.jpg', '.jpeg'} if extensions is None else set(ext.lower() for ext in extensions)
    root = Path(root_dir)

    for subfolder in root.iterdir():
        if not subfolder.is_dir():
            continue  # skip files at the top level

        kept, removed = 0, 0
        for file in subfolder.iterdir():
            # Only consider actual files
            if not file.is_file():
                continue

            if file.suffix.lower() in allowed:
                kept += 1
            else:
                file.unlink()   # delete non-image file
                removed += 1

        print(f"Subfolder '{subfolder.name}': kept {kept} images, removed {removed} non-images")

def split_dataset(data_dir, train_ratio, val_ratio, test_ratio):
    """
    Split each class‑folder under data_dir into train/validation/test
    according to the three ratios (which must add to 1.0).
    """
    # Validate ratios sum to 1
    if train_ratio + val_ratio + test_ratio != 1.0:
        print("Error: train_ratio + val_ratio + test_ratio must equal 1.0")
        return

    # Find class folders skipping any existing split directories
    classes = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
           and d not in ('train', 'validation', 'test')
    ]

    # Create train/validation/test subfolders for each class
    for split in ('train', 'validation', 'test'):
        for cls in classes:
            os.makedirs(os.path.join(data_dir, split, cls), exist_ok=True)

    # Shuffle and move
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        files = sorted(os.listdir(cls_path)) # Sorted for reproducibility
        random.shuffle(files)

        total = len(files)
        n_train = int(total * train_ratio)
        n_val   = int(total * val_ratio)
        n_test  = int(total * test_ratio)

        for i, fname in enumerate(files):
            src = os.path.join(cls_path, fname)

            if i < n_train:
                split = 'train'
            elif i < n_train + n_val:
                split = 'validation'
            elif i < n_train + n_val + n_test:
                split = 'test'
            else:
                # In case of any rounding leftovers, put them in 'test'
                split = 'test'

            dst = os.path.join(data_dir, split, cls, fname)
            shutil.move(src, dst)

        # Remove empty original folder
        os.rmdir(cls_path)

        # Feedback exact counts
        print(f"Class '{cls}': train={n_train}, validation={n_val}, test={n_test}")
    
def fetch_kaggle_dataset(kaggle_path, dest_folder):
    """
    Download and extract a dataset from Kaggle.

    This function uses the Kaggle API to download the specified dataset
    as a ZIP file into `dest_folder`, then unpacks it and removes the ZIP.

    Args:
        kaggle_path (str): The Kaggle dataset identifier, e.g. "user/dataset-name".
        dest_folder (str or Path): Local directory where the dataset will be downloaded and extracted.

    Raises:
        OSError: If authentication fails or the Kaggle API cannot access the dataset.
    """
    import zipfile
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    api.dataset_download_files(kaggle_path, path=dest_folder, unzip=False)
    with zipfile.ZipFile(f"{dest_folder}/{kaggle_path.split('/')[-1]}.zip","r") as z:
        z.extractall(dest_folder)
    os.remove(f"{dest_folder}/{kaggle_path.split('/')[-1]}.zip")
