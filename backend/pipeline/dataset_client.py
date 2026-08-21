import os
import json
from pathlib import Path

def fetch_msmarco_xi_dataset(max_records=100):
    print("Loading dataset...")
    base_dir = Path(__file__).resolve().parent.parent
    
    filename = os.getenv("DATASET_FILENAME", "msmarco_xi.json")
    dataset_file = base_dir / "data" / filename
    mock_file = base_dir / "data" / "mock_dataset.json"
    
    target_file = dataset_file if dataset_file.exists() else mock_file
    
    if target_file.exists():
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"[KB STARTUP] dataset_path = {target_file}")
            print(f"[KB STARTUP] dataset_exists = True")
            print(f"Successfully loaded {len(data)} records from {target_file.name}")
            return data[:max_records]
    else:
        print(f"[KB STARTUP] dataset_path = {mock_file}")
        print(f"[KB STARTUP] dataset_exists = False")
        raise FileNotFoundError(f"Mock dataset not found at {mock_file}. Cannot initialize knowledge base without data.")
