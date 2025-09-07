#!/usr/bin/env python3
"""
Fixed Natural Questions Dataset Processor for M1 MacBook Air
Handles numpy arrays and complex nested structures properly
"""

import os
import sys
import json
import logging
import gc
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Iterator, Optional
import pyarrow.parquet as pq
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/shivam/Documents/Shubham/Research project/logs/processing/natural_questions_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def safe_convert_to_json_serializable(obj, max_depth=10, current_depth=0):
    """Safely convert any object to JSON-serializable format with depth limit"""
    if current_depth > max_depth:
        return str(obj)  # Convert to string if too deep
    
    if obj is None or pd.isna(obj):
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): safe_convert_to_json_serializable(v, max_depth, current_depth + 1) 
                for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_convert_to_json_serializable(item, max_depth, current_depth + 1) 
                for item in obj]
    elif hasattr(obj, 'tolist'):  # pandas Series, numpy arrays, etc.
        try:
            return safe_convert_to_json_serializable(obj.tolist(), max_depth, current_depth + 1)
        except:
            return str(obj)
    elif hasattr(obj, 'to_dict'):  # pandas objects
        try:
            return safe_convert_to_json_serializable(obj.to_dict(), max_depth, current_depth + 1)
        except:
            return str(obj)
    else:
        return str(obj)  # Fallback to string representation


class FixedNaturalQuestionsProcessor:
    """Process Natural Questions dataset from Parquet files with robust error handling"""
    
    def __init__(self, 
                 raw_data_dir: str = "/Users/shivam/Documents/Shubham/Research project/data/raw/qa/natural_questions",
                 processed_data_dir: str = "/Users/shivam/Documents/Shubham/Research project/data/processed/qa",
                 max_memory_gb: float = 2.0,
                 batch_size: int = 500):  # Smaller batch size for safety
        
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.downloads_dir = self.raw_data_dir / "downloads"
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.batch_size = batch_size
        
        # Create output directory
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize stats
        self.stats = {
            'files_processed': 0,
            'examples_processed': 0,
            'errors': 0,
            'batches_saved': 0
        }
        
        self.logger = logger
        self.logger.info("Fixed Natural Questions processor initialized")
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits"""
        memory_usage = psutil.virtual_memory().used
        return memory_usage < self.max_memory_bytes
    
    def find_parquet_files(self) -> List[Path]:
        """Find all Parquet files in downloads directory"""
        parquet_files = []
        
        for file_path in self.downloads_dir.iterdir():
            if file_path.is_file() and not file_path.suffix and not file_path.name.endswith('.json') and not file_path.name.endswith('.lock'):
                try:
                    # Verify it's a Parquet file
                    pq.read_metadata(file_path)
                    parquet_files.append(file_path)
                except Exception:
                    continue
        
        return sorted(parquet_files)
    
    def process_parquet_file(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """Process a single Parquet file and yield examples"""
        try:
            parquet_file = pq.ParquetFile(file_path)
            
            for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                df = batch.to_pandas()
                
                for _, row in df.iterrows():
                    example = self.extract_qa_example(row)
                    if example:
                        yield example
                
                # Memory management
                del df
                if not self.check_memory_limit():
                    gc.collect()
                    
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            self.stats['errors'] += 1
    
    def extract_qa_example(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Extract QA example from a pandas row with robust error handling"""
        try:
            # Extract ID
            example_id = str(row.get('id', ''))
            if len(example_id) == 0:
                return None
            
            # Extract question text
            question_text = ''
            if 'question' in row and pd.notna(row['question']):
                question_data = row['question']
                if isinstance(question_data, dict) and 'text' in question_data:
                    question_text = str(question_data['text'])
            
            if len(question_text) == 0:
                return None
            
            # Extract document information with safe conversion
            document_title = ''
            document_url = ''
            document_tokens = []
            
            if 'document' in row and pd.notna(row['document']):
                document_data = row['document']
                if isinstance(document_data, dict):
                    document_title = str(document_data.get('title', ''))
                    document_url = str(document_data.get('url', ''))
                    
                    # Extract tokens safely
                    if 'tokens' in document_data:
                        tokens_data = document_data['tokens']
                        if isinstance(tokens_data, dict) and 'token' in tokens_data:
                            token_list = tokens_data['token']
                            if hasattr(token_list, '__iter__'):
                                try:
                                    # Convert to list and limit size
                                    document_tokens = list(token_list)[:50]  # Limit to 50 tokens
                                except:
                                    document_tokens = []
            
            # Extract annotations safely
            annotations = {}
            if 'annotations' in row and pd.notna(row['annotations']):
                try:
                    annotations_data = row['annotations']
                    # Safely convert annotations
                    annotations = safe_convert_to_json_serializable(annotations_data, max_depth=5)
                except Exception as e:
                    self.logger.warning(f"Error processing annotations: {e}")
                    annotations = {}
            
            # Build the example with safe conversion
            example = {
                'id': example_id,
                'question': question_text,
                'document_title': document_title,
                'document_url': document_url,
                'document_tokens': safe_convert_to_json_serializable(document_tokens),
                'annotations': annotations
            }
            
            # Final safety check - try to serialize
            try:
                json.dumps(example)
                return example
            except Exception as e:
                self.logger.warning(f"JSON serialization failed for example {example_id}: {e}")
                # Return a minimal safe version
                return {
                    'id': example_id,
                    'question': question_text,
                    'document_title': document_title,
                    'document_url': document_url,
                    'document_tokens': [],
                    'annotations': {}
                }
            
        except Exception as e:
            self.logger.warning(f"Error extracting example: {e}")
        
        return None
    
    def save_processed_batch(self, examples: List[Dict[str, Any]], batch_num: int) -> str:
        """Save a batch of processed examples"""
        output_file = self.processed_data_dir / f"natural_questions_batch_{batch_num:04d}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        self.stats['batches_saved'] += 1
        return str(output_file)
    
    def process_all_files(self, test_mode: bool = False) -> Dict[str, Any]:
        """Process all Natural Questions Parquet files"""
        start_time = datetime.now()
        self.logger.info("Starting Fixed Natural Questions processing...")
        
        # Find all Parquet files
        parquet_files = self.find_parquet_files()
        total_files = len(parquet_files)
        
        if test_mode:
            parquet_files = parquet_files[:3]  # Process only first 3 files in test mode
            self.logger.info(f"TEST MODE: Processing {len(parquet_files)} files")
        
        self.logger.info(f"Found {total_files} Parquet files")
        
        current_batch = []
        batch_num = 1
        
        for i, file_path in enumerate(parquet_files, 1):
            self.logger.info(f"Processing {file_path.name}")
            
            try:
                for example in self.process_parquet_file(file_path):
                    current_batch.append(example)
                    self.stats['examples_processed'] += 1
                    
                    # Save batch when it reaches batch_size
                    if len(current_batch) >= self.batch_size:
                        self.save_processed_batch(current_batch, batch_num)
                        self.logger.info(f"Saved batch {batch_num} with {len(current_batch)} examples")
                        current_batch = []
                        batch_num += 1
                
                self.stats['files_processed'] += 1
                self.logger.info(f"Completed file {i}/{len(parquet_files)}: {file_path.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                self.stats['errors'] += 1
        
        # Save remaining examples
        if current_batch:
            self.save_processed_batch(current_batch, batch_num)
            self.logger.info(f"Saved final batch {batch_num} with {len(current_batch)} examples")
        
        # Save summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            'total_files_found': total_files,
            'files_processed': self.stats['files_processed'],
            'examples_processed': self.stats['examples_processed'],
            'batches_saved': self.stats['batches_saved'],
            'errors': self.stats['errors'],
            'duration_seconds': duration,
            'output_directory': str(self.processed_data_dir)
        }
        
        summary_file = self.processed_data_dir / "natural_questions_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Processing complete! Summary: {summary}")
        return summary


def main():
    """Main processing function"""
    processor = FixedNaturalQuestionsProcessor()
    
    # Run in test mode first
    test_mode = len(sys.argv) > 1 and sys.argv[1] == '--test'
    
    try:
        summary = processor.process_all_files(test_mode=test_mode)
        
        print(f"\n{'='*60}")
        print("NATURAL QUESTIONS PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total files found: {summary['total_files_found']}")
        print(f"Files processed: {summary['files_processed']}")
        print(f"Examples processed: {summary['examples_processed']}")
        print(f"Batches saved: {summary['batches_saved']}")
        print(f"Errors: {summary['errors']}")
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        print(f"Output directory: {summary['output_directory']}")
        print("Natural Questions processing completed!")
        print("Check data/processed/qa/ for output files")
        print("Check logs/processing/natural_questions_processing.log for detailed logs")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
