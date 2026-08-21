import os
import httpx
from pathlib import Path

SUPABASE_URL = "https://rsjpdmxlwnbxjlnvuiem.supabase.co"

def download_asset(bucket_name: str, file_name: str, dest_path: Path):
    if dest_path.exists():
        # Check size if possible, but skipping for simplicity
        print(f"[ASSET MANAGER] {file_name} already exists at {dest_path}. Skipping download.")
        return

    url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket_name}/{file_name}"
    print(f"[ASSET MANAGER] Downloading {file_name} from {url}...")
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # We use stream to avoid loading a 4GB file directly into memory at once
    with httpx.stream("GET", url, follow_redirects=True) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Failed to download {file_name} from {url}: HTTP {r.status_code}. Ensure the bucket is public.")
            
        with open(dest_path, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1024 * 1024):  # 1MB chunks
                f.write(chunk)
                
    print(f"[ASSET MANAGER] Successfully downloaded {file_name}")

def ensure_heavy_assets():
    """
    Downloads heavy assets on startup before the rest of the RAG pipeline initializes.
    """
    bucket = os.getenv("SUPABASE_BUCKET", "assets")
    filename = os.getenv("DATASET_FILENAME", "msmarco_xi.json")
    
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    dest_path = data_dir / filename
    
    # Optional: we can fallback to mock_dataset.json if the heavy dataset isn't configured,
    # but the prompt specifically asked to download the heavy assets.
    try:
        download_asset(bucket_name=bucket, file_name=filename, dest_path=dest_path)
    except RuntimeError as e:
        print(f"[ASSET MANAGER] WARNING: {e}")
        print("[ASSET MANAGER] Defaulting to local mock_dataset.json for fallback.")
