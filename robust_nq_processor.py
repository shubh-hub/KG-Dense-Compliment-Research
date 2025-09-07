#!/usr/bin/env python3
"""
Robust Natural Questions Dataset Processor for M1 MacBook Air
Handles all 294 Parquet files with proper numpy array handling
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


def safe_str_conversion(obj):
    """Safely convert any object to string, handling numpy arrays"""
    if obj is None or pd.isna(obj):
        return ''
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, (int, float)):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        if len(obj) == 0:
            return ''
        elif len(obj) == 1:
            return str(obj[0])
        else:
            return ' '.join(str(x) for x in obj[:10])  # First 10 items
    elif isinstance(obj, dict) and 'text' in obj:
        return safe_str_conversion(obj['text'])
    else:
        return str(obj)


def safe_list_conversion(obj, max_items=50):
    """Safely convert numpy arrays to Python lists with size limit"""
    if obj is None or pd.isna(obj):
        return []
    elif isinstance(obj, np.ndarray):
        return obj[:max_items].tolist()
    elif isinstance(obj, list):
        return obj[:max_items]
    elif isinstance(obj, dict) and 'token' in obj:
        return safe_list_conversion(obj['token'], max_items)
    else:
        return []


def safe_dict_conversion(obj, max_depth=3, current_depth=0):
    """Safely convert nested structures to JSON-serializable format"""
    if current_depth > max_depth:
        return str(obj)
    
    if obj is None or pd.isna(obj):
        return {}
    elif isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if isinstance(value, np.ndarray):
                if value.dtype == 'object' and len(value) > 0:
                    result[key] = value[:10].tolist()  # Limit array size
                elif len(value) > 0:
                    result[key] = value[:10].tolist()
                else:
                    result[key] = []
            elif isinstance(value, (list, tuple)):
                result[key] = list(value)[:10]  # Limit list size
            elif isinstance(value, dict):
                result[key] = safe_dict_conversion(value, max_depth, current_depth + 1)
            else:
                result[key] = value
        return result
    else:
        return {}


class RobustNaturalQuestionsProcessor:
    """Robust processor for Natural Questions dataset handling all edge cases"""
    
    def __init__(self, 
                 raw_data_dir: str = "/Users/shivam/Documents/Shubham/Research project/data/raw/qa/natural_questions",
                 processed_data_dir: str = "/Users/shivam/Documents/Shubham/Research project/data/processed/qa",
                 max_memory_gb: float = 2.0,
                 batch_size: int = 200):  # Smaller batch for safety
        
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
            'batches_saved': 0,
            'skipped_examples': 0
        }
        
        self.logger = logger
        self.logger.info("Robust Natural Questions processor initialized")
    
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
        """Extract QA example from pandas row with robust numpy handling"""
        try:
            # Extract ID safely
            example_id = safe_str_conversion(row.get('id', ''))
            if len(example_id) == 0:
                self.stats['skipped_examples'] += 1
                return None
            
            # Extract question text safely
            question_data = row.get('question', {})
            question_text = safe_str_conversion(question_data)
            
            if len(question_text) == 0:
                self.stats['skipped_examples'] += 1
                return None
            
            # Extract document information safely
            document_data = row.get('document', {})
            document_title = ''
            document_url = ''
            document_tokens = []
            
            if isinstance(document_data, dict):
                document_title = safe_str_conversion(document_data.get('title', ''))
                document_url = safe_str_conversion(document_data.get('url', ''))
                
                # Extract tokens safely
                tokens_data = document_data.get('tokens', {})
                if isinstance(tokens_data, dict):
                    document_tokens = safe_list_conversion(tokens_data.get('token', []), max_items=50)
            
            # Extract annotations safely (simplified to avoid complexity)
            annotations_data = row.get('annotations', {})
            annotations = safe_dict_conversion(annotations_data, max_depth=2)
            
            # Build the example
            example = {
                'id': example_id,
                'question': question_text,
                'document_title': document_title,
                'document_url': document_url,
                'document_tokens': document_tokens,
                'annotations': annotations
            }
            
            # Final validation - ensure JSON serializable
            try:
                json.dumps(example)
                return example
            except Exception as e:
                self.logger.warning(f"JSON serialization failed for {example_id}: {e}")
                # Return minimal safe version
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
            self.stats['skipped_examples'] += 1
        
        return None
    
    def save_processed_batch(self, examples: List[Dict[str, Any]], batch_num: int) -> str:
        """Save a batch of processed examples"""
        output_file = self.processed_data_dir / f"natural_questions_batch_{batch_num:04d}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        self.stats['batches_saved'] += 1
        return str(output_file)
    
    def process_all_files(self, max_files: Optional[int] = None) -> Dict[str, Any]:
        """Process all Natural Questions Parquet files"""
        start_time = datetime.now()
        self.logger.info("Starting Robust Natural Questions processing...")
        
        # Find all Parquet files
        parquet_files = self.find_parquet_files()
        total_files = len(parquet_files)
        
        if max_files:
            parquet_files = parquet_files[:max_files]
            self.logger.info(f"Processing {len(parquet_files)} files (limited from {total_files})")
        else:
            self.logger.info(f"Processing all {total_files} Parquet files")
        
        current_batch = []
        batch_num = 1
        
        for i, file_path in enumerate(parquet_files, 1):
            self.logger.info(f"Processing file {i}/{len(parquet_files)}: {file_path.name}")
            
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
                
                # Memory check after each file
                if not self.check_memory_limit():
                    self.logger.warning("Memory limit approaching, forcing garbage collection")
                    gc.collect()
                
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
            'skipped_examples': self.stats['skipped_examples'],
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
    processor = RobustNaturalQuestionsProcessor()
    
    # Check command line arguments
    test_mode = len(sys.argv) > 1 and sys.argv[1] == '--test'
    max_files = 10 if test_mode else None
    
    try:
        summary = processor.process_all_files(max_files=max_files)
        
        print(f"\n{'='*60}")
        print("ROBUST NATURAL QUESTIONS PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total files found: {summary['total_files_found']}")
        print(f"Files processed: {summary['files_processed']}")
        print(f"Examples processed: {summary['examples_processed']}")
        print(f"Skipped examples: {summary['skipped_examples']}")
        print(f"Batches saved: {summary['batches_saved']}")
        print(f"Errors: {summary['errors']}")
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        print(f"Output directory: {summary['output_directory']}")
        print("Natural Questions processing completed!")
        print("Check data/processed/qa/ for output files")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
