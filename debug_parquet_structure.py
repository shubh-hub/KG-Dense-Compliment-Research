#!/usr/bin/env python3
"""
Debug Natural Questions Parquet file structure
"""

import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

def examine_parquet_file(file_path):
    """Examine the structure of a Parquet file"""
    print(f"Examining: {file_path.name}")
    print("="*50)
    
    try:
        # Read metadata
        parquet_file = pq.ParquetFile(file_path)
        metadata = parquet_file.metadata
        
        print(f"Number of rows: {metadata.num_rows}")
        print(f"Number of columns: {metadata.num_columns}")
        print(f"Number of row groups: {metadata.num_row_groups}")
        
        # Get schema
        schema = parquet_file.schema_arrow
        print(f"\nSchema:")
        for i, field in enumerate(schema):
            print(f"  {i}: {field.name} ({field.type})")
        
        # Read first few rows to see actual data
        print(f"\nFirst 3 rows:")
        df = pq.read_table(file_path).to_pandas()
        print(df.head(3))
        
        print(f"\nColumn names: {list(df.columns)}")
        print(f"Data types:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        
        # Check for specific columns we're looking for
        expected_cols = ['example_id', 'question_text', 'document_title', 'document_url', 'annotations', 'document_tokens']
        found_cols = []
        for col in expected_cols:
            if col in df.columns:
                found_cols.append(col)
        
        print(f"\nExpected columns found: {found_cols}")
        print(f"Missing columns: {[col for col in expected_cols if col not in found_cols]}")
        
        # Show sample data for key columns
        if 'question_text' in df.columns:
            print(f"\nSample questions:")
            for i, q in enumerate(df['question_text'].head(3)):
                print(f"  {i+1}: {q}")
        
        return df.columns.tolist(), len(df)
        
    except Exception as e:
        print(f"Error examining file: {e}")
        return [], 0

def main():
    downloads_dir = Path("/Users/shivam/Documents/Shubham/Research project/data/raw/qa/natural_questions/downloads")
    
    # Find first few Parquet files
    parquet_files = []
    for file_path in downloads_dir.iterdir():
        if file_path.is_file() and not file_path.suffix and not file_path.name.endswith('.json') and not file_path.name.endswith('.lock'):
            try:
                pq.read_metadata(file_path)
                parquet_files.append(file_path)
                if len(parquet_files) >= 3:  # Examine first 3 files
                    break
            except Exception:
                continue
    
    print(f"Found {len(parquet_files)} Parquet files to examine\n")
    
    all_columns = set()
    total_rows = 0
    
    for file_path in parquet_files:
        columns, rows = examine_parquet_file(file_path)
        all_columns.update(columns)
        total_rows += rows
        print("\n" + "="*70 + "\n")
    
    print(f"SUMMARY:")
    print(f"All unique columns across files: {sorted(all_columns)}")
    print(f"Total rows examined: {total_rows}")

if __name__ == "__main__":
    main()
