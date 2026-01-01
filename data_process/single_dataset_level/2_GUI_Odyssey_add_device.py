import json
import os
from tqdm import tqdm

def process_and_split_by_device():
    """
    Splits a large JSONL dataset into multiple files based on the device name 
    extracted from original annotation files.
    """
    # --- Path Configuration ---
    # Directory containing original individual annotation JSON files
    # (used to build the episode_id -> device_name mapping)
    anno_dir = './data/raw_annotations'
    
    # The source JSONL file to be split
    input_jsonl = "./datasets/source_data.jsonl"
    
    # Base directory for split output files
    output_base_dir = "./datasets/split_by_device"
    
    os.makedirs(output_base_dir, exist_ok=True)

    # --- Step 1: Build Metadata Mapping in Memory ---
    anno_files = [f for f in os.listdir(anno_dir) if f.endswith('.json')]
    id_to_device_map = {}
    
    print(f"Extracting device metadata from {len(anno_files)} annotation files...")
    for filename in tqdm(anno_files, desc="Building Mapping"):
        file_path = os.path.join(anno_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                eid = str(data.get('episode_id'))
                # Extract device name from the nested device_info dictionary
                device = data.get('device_info', {}).get('device_name', 'unknown_device')
                if eid:
                    id_to_device_map[eid] = device
            except Exception as e:
                print(f"Warning: Failed to parse {filename}: {e}")

    # --- Step 2: Stream and Split JSONL ---
    file_handles = {}
    print(f"\nProcessing {input_jsonl} and categorizing by device...")
    
    try:
        with open(input_jsonl, 'r', encoding='utf-8') as f:
            # Monitor progress line by line
            for line in tqdm(f, desc="Splitting JSONL"):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    eid = str(data.get("episode_id"))
                    
                    # Retrieve device name and sanitize it for file systems
                    device_name = id_to_device_map.get(eid, "unknown_device")
                    safe_device_name = device_name.replace(" ", "_")
                    
                    # Open a new file handle if this device hasn't been encountered yet
                    if safe_device_name not in file_handles:
                        out_path = os.path.join(output_base_dir, f"{safe_device_name}.jsonl")
                        # Use 'w' to overwrite existing or create new
                        file_handles[safe_device_name] = open(out_path, 'w', encoding='utf-8')
                    
                    # Write the entry to the corresponding device file
                    file_handles[safe_device_name].write(json.dumps(data, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Skipping corrupted line: {e}")

    finally:
        # Crucial: Ensure all open file handles are closed properly
        for fh in file_handles.values():
            fh.close()

    # --- Execution Summary ---
    print(f"\nTask completed successfully!")
    print(f"Output directory: {output_base_dir}")
    print("-" * 30)
    for device in sorted(file_handles.keys()):
        print(f"Category: {device.ljust(25)} -> Generated")

if __name__ == "__main__":
    process_and_split_by_device()