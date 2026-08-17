import requests
import json
import os

def fetch_msmarco_xi_dataset(max_records=100):
    """
    Fetches the ai4bharat/MSMARCO-XI dataset via the HuggingFace datasets-server REST API.
    Since the HF server may fail (e.g., ArrowNotImplementedError for this dataset),
    this function falls back to the local mock dataset on failure.
    """
    print(f"Fetching dataset ai4bharat/MSMARCO-XI ('hi' split) using datasets library...")
    try:
        from datasets import load_dataset
        # Load the default training data (contains Hindi and other languages)
        dataset = load_dataset("ai4bharat/MSMARCO-XI", "default", split="train", streaming=True)
        
        # Take the first `max_records`
        records = []
        for i, example in enumerate(dataset):
            if i >= max_records:
                break
            # Convert HuggingFace dataset example dict to a standard dict
            records.append(dict(example))
            
        if records:
            print(f"Successfully fetched {len(records)} records from HuggingFace.")
            return records
    except Exception as e:
        print(f"Error fetching from HuggingFace datasets: {e}")
        
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
