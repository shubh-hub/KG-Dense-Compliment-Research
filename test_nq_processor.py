#!/usr/bin/env python3
"""
Test the corrected Natural Questions processor on a single file
"""

import sys
sys.path.append('/Users/shivam/Documents/Shubham/Research project')

from natural_questions_processor import NaturalQuestionsProcessor
from pathlib import Path

def test_single_file():
    """Test processing a single Natural Questions file"""
    processor = NaturalQuestionsProcessor(batch_size=100)
    
    # Find first Parquet file
    downloads_dir = Path("/Users/shivam/Documents/Shubham/Research project/data/raw/qa/natural_questions/downloads")
    
    parquet_files = []
    for file_path in downloads_dir.iterdir():
        if file_path.is_file() and not file_path.suffix and not file_path.name.endswith('.json') and not file_path.name.endswith('.lock'):
            try:
                import pyarrow.parquet as pq
                pq.read_metadata(file_path)
                parquet_files.append(file_path)
                break  # Just test first file
            except Exception:
                continue
    
    if not parquet_files:
        print("No Parquet files found!")
        return
    
    test_file = parquet_files[0]
    print(f"Testing file: {test_file.name}")
    
    examples = []
    try:
        for example in processor.process_parquet_file(test_file):
            examples.append(example)
            if len(examples) >= 5:  # Just test first 5 examples
                break
        
        print(f"Successfully extracted {len(examples)} examples")
        
        if examples:
            print("\nSample example:")
            example = examples[0]
            print(f"ID: {example['id']}")
            print(f"Question: {example['question'][:100]}...")
            print(f"Document Title: {example['document_title']}")
            print(f"Document URL: {example['document_url'][:50]}...")
            print(f"Tokens: {len(example['document_tokens'])} tokens")
            print(f"Annotations keys: {list(example['annotations'].keys())}")
            
            # Test JSON serialization
            import json
            json_str = json.dumps(example)
            print(f"JSON serialization: SUCCESS ({len(json_str)} chars)")
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_single_file()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
