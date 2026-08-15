import requests
import json
import os

def fetch_msmarco_xi_dataset(max_records=100):
    """
    Fetches the ai4bharat/MSMARCO-XI dataset via the HuggingFace datasets-server REST API.
    Since the HF server may fail (e.g., ArrowNotImplementedError for this dataset),
    this function falls back to the local mock dataset on failure.
    """
    url = "https://datasets-server.huggingface.co/first-rows?dataset=ai4bharat%2FMSMARCO-XI&config=default&split=train"
    
    print(f"Fetching dataset from {url} ...")
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            rows = [r["row"] for r in data.get("rows", [])]
            if rows:
                print(f"Successfully fetched {len(rows)} rows from HF datasets-server.")
                return rows[:max_records]
        else:
            print(f"HF datasets-server returned {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Error fetching from HF datasets-server: {e}")
        
    print("Falling back to local mock dataset...")
    # Fallback to local mock dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mock_file = os.path.join(base_dir, "data", "mock_dataset.json")
    
    if os.path.exists(mock_file):
        with open(mock_file, "r", encoding="utf-8") as f:
            mock_data = json.load(f)
            return mock_data[:max_records]
    else:
        print(f"Mock dataset not found at {mock_file}. Please run data/mock_msmarco.py first.")
        return []
