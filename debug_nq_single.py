#!/usr/bin/env python3
"""
Debug a single Natural Questions example to understand the numpy array issue
"""

import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import json
from pathlib import Path

def examine_single_example():
    """Examine a single example to understand the data structure"""
    downloads_dir = Path("/Users/shivam/Documents/Shubham/Research project/data/raw/qa/natural_questions/downloads")
    
    # Find first Parquet file
    for file_path in downloads_dir.iterdir():
        if file_path.is_file() and not file_path.suffix and not file_path.name.endswith('.json') and not file_path.name.endswith('.lock'):
            try:
                pq.read_metadata(file_path)
                break
            except Exception:
                continue
    
    print(f"Examining file: {file_path.name}")
    
    # Read just first row
    df = pq.read_table(file_path).to_pandas()
    row = df.iloc[0]
    
    print("Raw row data types:")
    for col in df.columns:
        print(f"  {col}: {type(row[col])} - {row[col].__class__}")
    
    print(f"\nQuestion data: {type(row['question'])}")
    print(f"Question content: {row['question']}")
    
    print(f"\nDocument data: {type(row['document'])}")
    print(f"Document keys: {row['document'].keys() if hasattr(row['document'], 'keys') else 'No keys'}")
    
    if hasattr(row['document'], 'keys') and 'tokens' in row['document']:
        tokens = row['document']['tokens']
        print(f"\nTokens data: {type(tokens)}")
        if hasattr(tokens, 'keys'):
            print(f"Tokens keys: {tokens.keys()}")
            for key in tokens.keys():
                token_val = tokens[key]
                print(f"  {key}: {type(token_val)} - length: {len(token_val) if hasattr(token_val, '__len__') else 'N/A'}")
                if hasattr(token_val, '__len__') and len(token_val) > 0:
                    print(f"    First item type: {type(token_val[0])}")
    
    print(f"\nAnnotations data: {type(row['annotations'])}")
    if hasattr(row['annotations'], 'keys'):
        print(f"Annotations keys: {row['annotations'].keys()}")

if __name__ == "__main__":
    examine_single_example()
