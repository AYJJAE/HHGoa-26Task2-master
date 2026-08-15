import json
from datasets import load_dataset, get_dataset_config_names

def inspect_dataset():
    dataset_name = "ai4bharat/MSMARCO-XI"
    print(f"Inspecting dataset: {dataset_name}")
    
    try:
        # Get configurations (languages)
        configs = get_dataset_config_names(dataset_name)
        print(f"Configurations/Languages found ({len(configs)}): {configs}")
        
        # Load a small subset of the first configuration (e.g., 'hi' if available, or first one)
        lang = 'hi' if 'hi' in configs else configs[0]
        print(f"\nLoading small subset of split 'train' for lang '{lang}'...")
        
        # Load streaming=True to just get the first few elements without downloading everything
        dataset = load_dataset(dataset_name, lang, split="train", streaming=True)
        
        # Print schema and first record
        print("\nFeatures (Schema):")
        if hasattr(dataset, 'features'):
             print(json.dumps({k: str(v) for k, v in dataset.features.items()}, indent=2))
        else:
            print("Features not available on IterableDataset before iterating. Fetching first record...")
            
        print("\nFirst Record:")
        first_record = next(iter(dataset))
        print(json.dumps(first_record, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Error inspecting dataset: {e}")

if __name__ == "__main__":
    inspect_dataset()
