import os
import sys
import zipfile
import tarfile
import urllib.request
from pathlib import Path

CAMVID_URLS = [
    "https://github.com/AlexeyAB/darknet/releases/download/camvid/CamVid.zip",
    "https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz"
]

def check_dataset_complete(root: Path) -> bool:
    """Checks whether train, val, and test splits with images are already present."""
    patterns = [
        root / "train" / "images",
        root / "train",
        root / "images" / "train"
    ]
    for p in patterns:
        if p.exists() and len(list(p.glob("*.png")) + list(p.glob("*.jpg"))) >= 350:
            return True
    return False

def download_file(url: str, dest_path: str):
    """Download a remote file with progress reporting."""
    print(f"Downloading from {url}...")
    def _reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        sys.stdout.write(f"\r  Progress: {percent}% ({count * block_size / (1024*1024):.1f} MB)")
        sys.stdout.flush()
    urllib.request.urlretrieve(url, dest_path, reporthook=_reporthook)
    print("\nDownload complete!")

def extract_archive(archive_path: str, extract_to: str):
    """Extract .zip or .tar.gz archive."""
    print(f"Extracting {archive_path} to {extract_to}...")
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith('.tgz') or archive_path.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    print("Extraction complete!")

def setup_camvid_dataset(data_root: str = "./data/CamVid"):
    """
    Ensure the CamVid dataset is verified and ready.
    """
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    # Check if dataset already exists in any valid layout
    if check_dataset_complete(root):
        train_count = len(list(root.glob("**/train/**/*.png")) + list(root.glob("**/train/**/*.jpg"))) // 2
        print(f"CamVid dataset verified at {root.resolve()}! Found {train_count} training image-mask pairs.")
        return str(root.resolve())

    print(f"CamVid dataset not found or incomplete in {root.resolve()}. Starting automatic setup...")

    for url in CAMVID_URLS:
        ext = ".zip" if url.endswith(".zip") else ".tgz"
        archive_file = root / f"camvid_temp{ext}"
        try:
            download_file(url, str(archive_file))
            extract_archive(str(archive_file), str(root))
            if archive_file.exists():
                archive_file.unlink()
            if check_dataset_complete(root):
                break
        except Exception as e:
            print(f"Failed handling archive from {url}: {e}")
            if archive_file.exists():
                archive_file.unlink()

    print(f"Dataset ready at {root.resolve()}")
    return str(root.resolve())

if __name__ == "__main__":
    setup_camvid_dataset()
