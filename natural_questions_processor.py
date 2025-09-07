#!/usr/bin/env python3
"""
Natural Questions Dataset Processor for M1 MacBook Air
Processes 57GB of Natural Questions Parquet files with memory optimization
"""

import os
import sys
import json
import logging
import gc
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Iterator, Optional
import pyarrow.parquet as pq
import pandas as pd
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

def safe_len_check(obj):
    """Safely check if object has content without triggering numpy array boolean ambiguity"""
    if obj is None:
        return False
    try:
        if pd.isna(obj):
            return False
    except (TypeError, ValueError):
        # pd.isna can fail on some objects, continue with other checks
        pass
    
    if isinstance(obj, str):
        return len(obj) > 0
    elif isinstance(obj, (list, tuple)):
        return len(obj) > 0
    elif isinstance(obj, np.ndarray):
        return obj.size > 0
    elif isinstance(obj, dict):
        return len(obj) > 0
    elif isinstance(obj, (np.integer, np.floating)):
        return True  # Numbers are always "present"
    elif hasattr(obj, '__len__'):
        try:
            return len(obj) > 0
        except (TypeError, ValueError):
            return True  # If len fails, assume it exists
    else:
        # For any other type, assume it exists if it's not None
        return True

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if obj is None:
        return None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):
        # Handle pandas/pyarrow arrays and other array-like objects
        try:
            return convert_numpy_types(obj.tolist())
        except (AttributeError, TypeError):
            pass
    elif hasattr(obj, '__dict__'):
        # Handle objects with attributes
        try:
            return convert_numpy_types(obj.__dict__)
        except (AttributeError, TypeError):
            pass
    
    # Handle pandas NaN safely
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    
    # For any remaining numpy types, try to convert to Python native
    if hasattr(obj, 'dtype') and hasattr(obj, 'item'):
        try:
            return obj.item()  # Convert numpy scalar to Python scalar
        except (AttributeError, TypeError, ValueError):
            pass
    
    return obj


class NaturalQuestionsProcessor:
    """Process Natural Questions dataset from Parquet files with M1 optimization"""
    
    def __init__(self, 
                 raw_data_dir: str = "/Users/shivam/Documents/Shubham/Research project/data/raw/qa/natural_questions",
                 processed_data_dir: str = "/Users/shivam/Documents/Shubham/Research project/data/processed/qa",
                 max_memory_gb: float = 2.0,
                 batch_size: int = 1000):
        
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.downloads_dir = self.raw_data_dir / "downloads"
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.batch_size = batch_size
        
        # Create output directory
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logger
        
        # Statistics tracking
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_examples': 0,
            'processed_examples': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024 * 1024)
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits"""
        current_memory = self.get_memory_usage()
        if current_memory > self.max_memory_bytes / (1024 * 1024 * 1024):
            logger.warning(f"Memory usage ({current_memory:.2f}GB) approaching limit")
            gc.collect()
            return False
        return True
    
    def find_parquet_files(self) -> List[Path]:
        """Find all Parquet files in downloads directory"""
        parquet_files = []
        
        # Look for files without extensions (these are the actual Parquet files)
        for file_path in self.downloads_dir.iterdir():
            if file_path.is_file() and not file_path.suffix and not file_path.name.endswith('.json') and not file_path.name.endswith('.lock'):
                # Check if it's a Parquet file by trying to read metadata
                try:
                    pq.read_metadata(file_path)
                    parquet_files.append(file_path)
                except Exception:
                    continue
        
        logger.info(f"Found {len(parquet_files)} Parquet files")
        return sorted(parquet_files)
    
    def process_parquet_file(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """Process a single Parquet file and yield examples"""
        try:
            logger.info(f"Processing {file_path.name}")
            
            # Read Parquet file in batches
            parquet_file = pq.ParquetFile(file_path)
            
            for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                df = batch.to_pandas()
                
                for _, row in df.iterrows():
                    # Extract relevant fields for QA task
                    example = self.extract_qa_example(row)
                    if example:
                        yield example
                
                # Memory management
                del df
                if not self.check_memory_limit():
                    gc.collect()
                    
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            self.stats['errors'] += 1
    
    def extract_qa_example(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Extract QA example from a pandas row with correct Natural Questions structure"""
        try:
            # Extract ID (direct field)
            example_id = str(row.get('id', ''))
            if not safe_len_check(example_id):
                return None
            
            # Extract question (nested in 'question' struct)
            question_text = ''
            if 'question' in row and pd.notna(row['question']):
                question_data = row['question']
                if isinstance(question_data, dict) and 'text' in question_data:
                    question_text = str(question_data['text'])
            
            if not safe_len_check(question_text):
                return None
            
            # Extract document information (nested in 'document' struct)
            document_title = ''
            document_url = ''
            document_tokens = []
            
            if 'document' in row and pd.notna(row['document']):
                document_data = row['document']
                if isinstance(document_data, dict):
                    document_title = str(document_data.get('title', ''))
                    document_url = str(document_data.get('url', ''))
                    
                    # Extract tokens safely using safe_len_check
                    if 'tokens' in document_data and isinstance(document_data['tokens'], dict):
                        token_data = document_data['tokens']
                        if 'token' in token_data and safe_len_check(token_data['token']):
                            token_array = token_data['token']
                            # Convert numpy array to list safely
                            if isinstance(token_array, np.ndarray):
                                document_tokens = token_array[:100].tolist()  # Limit and convert
                            elif isinstance(token_array, list):
                                document_tokens = token_array[:100]
            
            # Extract annotations safely
            annotations = {}
            if 'annotations' in row and pd.notna(row['annotations']):
                annotations_data = row['annotations']
                if isinstance(annotations_data, dict):
                    annotations = convert_numpy_types(annotations_data)
            
            # Build the example
            example = {
                'id': example_id,
                'question': question_text,
                'document_title': document_title,
                'document_url': document_url,
                'document_tokens': document_tokens,
                'annotations': annotations
            }
            
            return example
            
        except Exception as e:
            self.logger.warning(f"Error extracting example: {e}")
        
        return None
    
    def save_processed_batch(self, examples: List[Dict[str, Any]], batch_num: int) -> str:
        """Save a batch of processed examples"""
        output_file = self.processed_data_dir / f"natural_questions_batch_{batch_num:04d}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                # Ensure all numpy types are converted before JSON serialization
                clean_example = convert_numpy_types(example)
                f.write(json.dumps(clean_example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved batch {batch_num} with {len(examples)} examples to {output_file}")
        return str(output_file)
    
    def process_all_files(self, max_examples: Optional[int] = None) -> Dict[str, Any]:
        """Process all Natural Questions Parquet files"""
        logger.info("Starting Natural Questions processing...")
        
        parquet_files = self.find_parquet_files()
        self.stats['total_files'] = len(parquet_files)
        
        if not parquet_files:
            logger.error("No Parquet files found!")
            return self.stats
        
        batch_examples = []
        batch_num = 1
        processed_files = []
        
        for file_path in parquet_files:
            try:
                self.stats['processed_files'] += 1
                
                for example in self.process_parquet_file(file_path):
                    batch_examples.append(example)
                    self.stats['processed_examples'] += 1
                    
                    # Save batch when it reaches batch_size
                    if len(batch_examples) >= self.batch_size:
                        output_file = self.save_processed_batch(batch_examples, batch_num)
                        processed_files.append(output_file)
                        batch_examples = []
                        batch_num += 1
                    
                    # Check if we've reached max examples
                    if max_examples and self.stats['processed_examples'] >= max_examples:
                        logger.info(f"Reached maximum examples limit: {max_examples}")
                        break
                
                logger.info(f"Completed file {self.stats['processed_files']}/{self.stats['total_files']}: {file_path.name}")
                
                if max_examples and self.stats['processed_examples'] >= max_examples:
                    break
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                self.stats['errors'] += 1
                continue
        
        # Save remaining examples
        if batch_examples:
            output_file = self.save_processed_batch(batch_examples, batch_num)
            processed_files.append(output_file)
        
        # Create summary file
        self.create_summary(processed_files)
        
        self.stats['end_time'] = datetime.now()
        self.stats['duration'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        logger.info(f"Natural Questions processing completed!")
        logger.info(f"Processed {self.stats['processed_examples']} examples from {self.stats['processed_files']} files")
        logger.info(f"Duration: {self.stats['duration']:.2f} seconds")
        
        return self.stats
    
    def create_summary(self, processed_files: List[str]):
        """Create summary of processed data"""
        summary = {
            'dataset': 'natural_questions',
            'processing_date': datetime.now().isoformat(),
            'stats': self.stats,
            'processed_files': processed_files,
            'total_examples': self.stats['processed_examples'],
            'format': 'jsonl',
            'source': 'huggingface_parquet'
        }
        
        summary_file = self.processed_data_dir / "natural_questions_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Summary saved to {summary_file}")

def main():
    """Main processing function"""
    processor = NaturalQuestionsProcessor()
    
    # For M1 MacBook, process a sample first to test
    logger.info("Processing Natural Questions dataset (M1 optimized)")
    
    # Process first 10,000 examples for testing
    stats = processor.process_all_files(max_examples=10000)
    
    print("\n" + "="*50)
    print("NATURAL QUESTIONS PROCESSING COMPLETE")
    print("="*50)
    print(f"Total files found: {stats['total_files']}")
    print(f"Files processed: {stats['processed_files']}")
    print(f"Examples processed: {stats['processed_examples']}")
    print(f"Errors: {stats['errors']}")
    print(f"Duration: {stats.get('duration', 0):.2f} seconds")
    print(f"Output directory: {processor.processed_data_dir}")
    
    return stats

if __name__ == "__main__":
    main()
