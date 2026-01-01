#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parquet to JSON Conversion Tool
Function: Converts Parquet storage files to JSON format with custom encoding 
          to handle NumPy types, bytes, and datetime objects.
"""

import pandas as pd
import json
import os
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def convert_parquet_to_json(parquet_file, output_file=None, orient='records'):
    """
    Converts a single Parquet file to JSON format.
    
    Args:
        parquet_file: Path to the input Parquet file.
        output_file: Path to the output JSON file.
        orient: JSON layout format, defaults to 'records'.
    """
    print(f"Processing: {parquet_file}")
    
    # Read Parquet file using pandas
    df = pd.read_parquet(parquet_file)
    
    if output_file is None:
        output_file = parquet_file.replace('.parquet', '.json')
    
    # Custom encoder to handle non-serializable types (NumPy, bytes, etc.)
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, bytes):
                try:
                    return obj.decode('utf-8')
                except UnicodeDecodeError:
                    # Fallback to base64 for non-UTF8 binary data
                    import base64
                    return base64.b64encode(obj).decode('ascii')
            if hasattr(obj, 'isoformat'):  # Handles datetime/timestamp objects
                return obj.isoformat()
            return super(NpEncoder, self).default(obj)
    
    data = df.to_dict(orient=orient)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=NpEncoder)
    
    print(f"Successfully converted to: {output_file}")
    print(f"Data shape: {df.shape}")
    return output_file

def convert_all_parquet_files(data_dir, output_dir=None, orient='records'):
    """
    Batch converts all Parquet files in a directory.
    """
    data_path = Path(data_dir)
    
    if output_dir is None:
        output_dir = data_path / "processed_json"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Locate all .parquet files
    parquet_files = list(data_path.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files. Starting conversion...")
    
    converted_files = []
    for parquet_file in tqdm(parquet_files, desc="Converting"):
        try:
            output_file = output_dir / f"{parquet_file.stem}.json"
            converted_file = convert_parquet_to_json(str(parquet_file), str(output_file), orient)
            converted_files.append(converted_file)
        except Exception as e:
            print(f"Error processing {parquet_file}: {e}")
    
    print(f"\nTask Complete! {len(converted_files)} files converted.")
    return converted_files

def preview_data(parquet_file, num_rows=5):
    """
    Simple preview to inspect data structure before conversion.
    """
    print(f"Previewing: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First {num_rows} rows:")
    print(df.head(num_rows))
    return df

def main():
    parser = argparse.ArgumentParser(description='Convert Parquet files to JSON format.')
    parser.add_argument('--data_dir', default='./data', help='Input directory containing parquet files.')
    parser.add_argument('--output_dir', help='Output directory for JSON files.')
    parser.add_argument('--orient', default='records', 
                        choices=['records', 'index', 'values', 'split', 'table'],
                        help='JSON orientation format.')
    parser.add_argument('--preview', action='store_true', help='Preview data without converting.')
    parser.add_argument('--single_file', help='Process a specific single file.')
    
    args = parser.parse_args()
    
    if args.preview:
        parquet_files = list(Path(args.data_dir).glob("*.parquet"))
        if parquet_files:
            preview_data(str(parquet_files[0]))
        else:
            print("No files to preview.")
    elif args.single_file:
        convert_parquet_to_json(args.single_file, orient=args.orient)
    else:
        convert_all_parquet_files(args.data_dir, args.output_dir, args.orient)

if __name__ == "__main__":
    main()