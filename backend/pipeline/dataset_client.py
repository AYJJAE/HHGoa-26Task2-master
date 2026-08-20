import json
import os

def fetch_msmarco_xi_dataset(max_records=100):
    print("Skipping HuggingFace and using local mock dataset directly...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mock_file = os.path.join(base_dir, "data", "mock_dataset.json")
    
    if os.path.exists(mock_file):
        with open(mock_file, "r", encoding="utf-8") as f:
            mock_data = json.load(f)
            print(f"Successfully loaded {len(mock_data)} records from local mock_dataset.json")
            return mock_data[:max_records]
    else:
        print(f"Mock dataset not found at {mock_file}.")
        return []
