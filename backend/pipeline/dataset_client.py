import json
from pathlib import Path

def fetch_msmarco_xi_dataset(max_records=100):
    print("Skipping HuggingFace and using local mock dataset directly...")
    base_dir = Path(__file__).resolve().parent.parent
    mock_file = base_dir / "data" / "mock_dataset.json"
    
    if mock_file.exists():
        with open(mock_file, "r", encoding="utf-8") as f:
            mock_data = json.load(f)
            print(f"[KB STARTUP] dataset_path = {mock_file}")
            print(f"[KB STARTUP] dataset_exists = True")
            print(f"Successfully loaded {len(mock_data)} records from local mock_dataset.json")
            return mock_data[:max_records]
    else:
        print(f"[KB STARTUP] dataset_path = {mock_file}")
        print(f"[KB STARTUP] dataset_exists = False")
        raise FileNotFoundError(f"Mock dataset not found at {mock_file}. Cannot initialize knowledge base without data.")
