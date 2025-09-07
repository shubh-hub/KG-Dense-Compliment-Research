#!/usr/bin/env python3
"""
Data Loaders and Preprocessing Pipelines for KG + Dense Vector Research
Handles QA, IR, and KG data loading with M1 optimization
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from collections import defaultdict
import random

logger = logging.getLogger(__name__)

class QADataset(Dataset):
    """Dataset for Question Answering tasks"""
    
    def __init__(self, data_path: str, tokenizer_name: str = "distilbert-base-uncased",
                 max_length: int = 512, kg_entities: Optional[Dict[str, int]] = None):
        self.data_path = Path(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.kg_entities = kg_entities or {}
        
        # Load data
        self.examples = self._load_data()
        logger.info(f"Loaded {len(self.examples)} QA examples from {data_path}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load QA data from JSONL file"""
        examples = []
        
        if not self.data_path.exists():
            logger.warning(f"Data file not found: {self.data_path}")
            return examples
        
        with open(self.data_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        example = json.loads(line)
                        examples.append(example)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {e}")
                        continue
        
        return examples
    
    def _extract_entities(self, text: str) -> List[int]:
        """Extract entity IDs from text (simplified approach)"""
        # This is a simplified entity linking approach
        # In practice, you'd use a proper entity linking system
        entities = []
        words = text.lower().split()
        
        for word in words:
            if word in self.kg_entities:
                entities.append(self.kg_entities[word])
        
        # Pad or truncate to fixed size
        max_entities = 10
        if len(entities) > max_entities:
            entities = entities[:max_entities]
        else:
            entities.extend([0] * (max_entities - len(entities)))
        
        return entities
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Get question and answer
        question = example.get('question', '')
        answers = example.get('short_answers', example.get('answers', []))
        
        # Create label (1 if has answer, 0 otherwise)
        label = 1 if answers and any(ans.strip() for ans in answers) else 0
        
        # Tokenize question
        encoding = self.tokenizer(
            question,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Extract entities
        entity_ids = self._extract_entities(question)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'entity_ids': torch.tensor(entity_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float),
            'example_id': example.get('example_id', f'qa_{idx}')
        }

class IRDataset(Dataset):
    """Dataset for Information Retrieval tasks"""
    
    def __init__(self, data_path: str, tokenizer_name: str = "distilbert-base-uncased",
                 max_length: int = 512, kg_entities: Optional[Dict[str, int]] = None):
        self.data_path = Path(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.kg_entities = kg_entities or {}
        
        # Load data
        self.examples = self._load_data()
        logger.info(f"Loaded {len(self.examples)} IR examples from {data_path}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load IR data from JSONL file"""
        examples = []
        
        if not self.data_path.exists():
            logger.warning(f"Data file not found: {self.data_path}")
            return examples
        
        with open(self.data_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        example = json.loads(line)
                        examples.append(example)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {e}")
                        continue
        
        return examples
    
    def _extract_entities(self, text: str) -> List[int]:
        """Extract entity IDs from text"""
        entities = []
        words = text.lower().split()
        
        for word in words:
            if word in self.kg_entities:
                entities.append(self.kg_entities[word])
        
        # Pad or truncate to fixed size
        max_entities = 10
        if len(entities) > max_entities:
            entities = entities[:max_entities]
        else:
            entities.extend([0] * (max_entities - len(entities)))
        
        return entities
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Get query and passage
        query = example.get('query', '')
        passage = example.get('passage', '')
        relevance = example.get('relevance', 0)
        
        # Combine query and passage for encoding
        combined_text = f"{query} [SEP] {passage}"
        
        # Tokenize
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Extract entities from query
        entity_ids = self._extract_entities(query)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'entity_ids': torch.tensor(entity_ids, dtype=torch.long),
            'relevance': torch.tensor(relevance, dtype=torch.float),
            'query_ids': example.get('query_id', f'q_{idx}'),
            'doc_ids': example.get('passage_id', f'd_{idx}')
        }

class KGDataset(Dataset):
    """Dataset for Knowledge Graph data"""
    
    def __init__(self, data_path: str, entity_to_id: Dict[str, int], 
                 relation_to_id: Dict[str, int]):
        self.data_path = Path(data_path)
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id
        
        # Load triples
        self.triples = self._load_triples()
        self.edge_index, self.edge_type = self._create_graph()
        
        logger.info(f"Loaded {len(self.triples)} KG triples from {data_path}")
    
    def _load_triples(self) -> List[Tuple[int, int, int]]:
        """Load KG triples from JSONL file"""
        triples = []
        
        if not self.data_path.exists():
            logger.warning(f"Data file not found: {self.data_path}")
            return triples
        
        with open(self.data_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        example = json.loads(line)
                        head = example.get('head', '')
                        relation = example.get('relation', '')
                        tail = example.get('tail', '')
                        
                        # Convert to IDs
                        head_id = self.entity_to_id.get(head, 0)
                        relation_id = self.relation_to_id.get(relation, 0)
                        tail_id = self.entity_to_id.get(tail, 0)
                        
                        if head_id > 0 and tail_id > 0 and relation_id > 0:
                            triples.append((head_id, relation_id, tail_id))
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {e}")
                        continue
        
        return triples
    
    def _create_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create graph structure from triples"""
        if not self.triples:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.long)
        
        edges = []
        edge_types = []
        
        for head, relation, tail in self.triples:
            edges.append([head, tail])
            edge_types.append(relation)
            
            # Add reverse edge for undirected graph
            edges.append([tail, head])
            edge_types.append(relation)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        return edge_index, edge_type
    
    def get_graph_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get graph structure for GNN"""
        return self.edge_index, self.edge_type
    
    def __len__(self) -> int:
        return len(self.triples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        head, relation, tail = self.triples[idx]
        
        return {
            'head': torch.tensor(head, dtype=torch.long),
            'relation': torch.tensor(relation, dtype=torch.long),
            'tail': torch.tensor(tail, dtype=torch.long)
        }

class DataProcessor:
    """Main data processor for creating datasets and dataloaders"""
    
    def __init__(self, data_dir: str, tokenizer_name: str = "distilbert-base-uncased"):
        self.data_dir = Path(data_dir)
        self.tokenizer_name = tokenizer_name
        
        # Load entity and relation mappings
        self.entity_to_id, self.relation_to_id = self._load_kg_mappings()
        
    def _load_kg_mappings(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Load entity and relation ID mappings"""
        entity_to_id = {}
        relation_to_id = {}
        
        # Try to load from metadata files
        kg_dir = self.data_dir / "processed" / "kg"
        
        if kg_dir.exists():
            # Load from processed KG files
            for kg_file in kg_dir.glob("*.jsonl"):
                with open(kg_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                example = json.loads(line)
                                head = example.get('head', '')
                                relation = example.get('relation', '')
                                tail = example.get('tail', '')
                                
                                if head and head not in entity_to_id:
                                    entity_to_id[head] = len(entity_to_id) + 1
                                if tail and tail not in entity_to_id:
                                    entity_to_id[tail] = len(entity_to_id) + 1
                                if relation and relation not in relation_to_id:
                                    relation_to_id[relation] = len(relation_to_id) + 1
                                    
                            except json.JSONDecodeError:
                                continue
        
        logger.info(f"Loaded {len(entity_to_id)} entities and {len(relation_to_id)} relations")
        return entity_to_id, relation_to_id
    
    def create_qa_dataset(self, split: str = "train", max_length: int = 512) -> QADataset:
        """Create QA dataset"""
        qa_files = list((self.data_dir / "processed" / "qa").glob(f"*{split}*.jsonl"))
        
        if not qa_files:
            # Try alternative naming
            qa_files = list((self.data_dir / "processed" / "qa").glob("*.jsonl"))
            if qa_files:
                qa_files = [qa_files[0]]  # Use first available file
        
        if not qa_files:
            raise FileNotFoundError(f"No QA data files found for split: {split}")
        
        return QADataset(
            data_path=str(qa_files[0]),
            tokenizer_name=self.tokenizer_name,
            max_length=max_length,
            kg_entities=self.entity_to_id
        )
    
    def create_ir_dataset(self, split: str = "test", max_length: int = 512) -> IRDataset:
        """Create IR dataset"""
        ir_files = list((self.data_dir / "processed" / "ir").glob(f"*{split}*.jsonl"))
        
        if not ir_files:
            # Try any IR file
            ir_files = list((self.data_dir / "processed" / "ir").glob("*.jsonl"))
            if ir_files:
                ir_files = [ir_files[0]]  # Use first available file
        
        if not ir_files:
            raise FileNotFoundError(f"No IR data files found for split: {split}")
        
        return IRDataset(
            data_path=str(ir_files[0]),
            tokenizer_name=self.tokenizer_name,
            max_length=max_length,
            kg_entities=self.entity_to_id
        )
    
    def create_kg_dataset(self, split: str = "train") -> KGDataset:
        """Create KG dataset"""
        kg_files = list((self.data_dir / "processed" / "kg").glob(f"*{split}*.jsonl"))
        
        if not kg_files:
            # Try any KG file
            kg_files = list((self.data_dir / "processed" / "kg").glob("*.jsonl"))
            if kg_files:
                kg_files = [kg_files[0]]  # Use first available file
        
        if not kg_files:
            raise FileNotFoundError(f"No KG data files found for split: {split}")
        
        return KGDataset(
            data_path=str(kg_files[0]),
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id
        )
    
    def create_dataloaders(self, batch_size: int = 16, num_workers: int = 2) -> Dict[str, DataLoader]:
        """Create all dataloaders"""
        dataloaders = {}
        
        try:
            # QA DataLoader
            qa_dataset = self.create_qa_dataset("train")
            dataloaders['qa_dataloader'] = DataLoader(
                qa_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=self._qa_collate_fn
            )
            logger.info(f"Created QA dataloader with {len(qa_dataset)} examples")
            
        except FileNotFoundError as e:
            logger.warning(f"Could not create QA dataloader: {e}")
        
        try:
            # IR DataLoader
            ir_dataset = self.create_ir_dataset("test")
            dataloaders['ir_dataloader'] = DataLoader(
                ir_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=self._ir_collate_fn
            )
            logger.info(f"Created IR dataloader with {len(ir_dataset)} examples")
            
        except FileNotFoundError as e:
            logger.warning(f"Could not create IR dataloader: {e}")
        
        try:
            # KG Dataset (not a dataloader, used for graph structure)
            kg_dataset = self.create_kg_dataset("train")
            dataloaders['kg_dataset'] = kg_dataset
            logger.info(f"Created KG dataset with {len(kg_dataset)} triples")
            
        except FileNotFoundError as e:
            logger.warning(f"Could not create KG dataset: {e}")
        
        return dataloaders
    
    def _qa_collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for QA batches"""
        collated = {}
        
        # Stack tensors
        for key in ['input_ids', 'attention_mask', 'entity_ids', 'labels']:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])
        
        # Keep example IDs as list
        if 'example_id' in batch[0]:
            collated['example_ids'] = [item['example_id'] for item in batch]
        
        # Add graph data if KG dataset available
        if hasattr(self, '_kg_edge_index'):
            collated['edge_index'] = self._kg_edge_index
            collated['edge_type'] = self._kg_edge_type
        
        return collated
    
    def _ir_collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for IR batches"""
        collated = {}
        
        # Stack tensors
        for key in ['input_ids', 'attention_mask', 'entity_ids', 'relevance']:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])
        
        # Keep IDs as lists
        for key in ['query_ids', 'doc_ids']:
            if key in batch[0]:
                collated[key] = [item[key] for item in batch]
        
        # Add graph data if KG dataset available
        if hasattr(self, '_kg_edge_index'):
            collated['edge_index'] = self._kg_edge_index
            collated['edge_type'] = self._kg_edge_type
        
        return collated
    
    def prepare_kg_graph_data(self):
        """Prepare KG graph data for models"""
        try:
            kg_dataset = self.create_kg_dataset("train")
            self._kg_edge_index, self._kg_edge_type = kg_dataset.get_graph_data()
            logger.info(f"Prepared KG graph with {self._kg_edge_index.shape[1]} edges")
        except FileNotFoundError:
            logger.warning("Could not prepare KG graph data")
            self._kg_edge_index = torch.empty((2, 0), dtype=torch.long)
            self._kg_edge_type = torch.empty(0, dtype=torch.long)

class DataConfig:
    """Configuration for data processing"""
    
    def __init__(self):
        self.tokenizer_name = "distilbert-base-uncased"
        self.max_length = 512
        self.batch_size = 16
        self.num_workers = 2
        self.data_dir = "data"
        
    @classmethod
    def lightweight_config(cls):
        """Lightweight config for M1 MacBook"""
        config = cls()
        config.tokenizer_name = "distilbert-base-uncased"
        config.max_length = 256
        config.batch_size = 8
        config.num_workers = 1
        return config
    
    @classmethod
    def standard_config(cls):
        """Standard config for regular training"""
        config = cls()
        config.tokenizer_name = "bert-base-uncased"
        config.max_length = 512
        config.batch_size = 16
        config.num_workers = 2
        return config

def create_data_processor(config: DataConfig = None) -> DataProcessor:
    """Create data processor with configuration"""
    if config is None:
        config = DataConfig.lightweight_config()
    
    processor = DataProcessor(
        data_dir=config.data_dir,
        tokenizer_name=config.tokenizer_name
    )
    
    # Prepare KG graph data
    processor.prepare_kg_graph_data()
    
    return processor

if __name__ == "__main__":
    # Test data loading
    print("Testing data loaders...")
    
    # Create lightweight processor for testing
    config = DataConfig.lightweight_config()
    processor = create_data_processor(config)
    
    try:
        # Create dataloaders
        dataloaders = processor.create_dataloaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        
        print(f"Created {len(dataloaders)} dataloaders:")
        for name, loader in dataloaders.items():
            if isinstance(loader, DataLoader):
                print(f"  {name}: {len(loader)} batches")
            else:
                print(f"  {name}: {len(loader)} items")
        
        # Test a batch if available
        if 'qa_dataloader' in dataloaders:
            batch = next(iter(dataloaders['qa_dataloader']))
            print(f"QA batch keys: {list(batch.keys())}")
            print(f"QA batch size: {batch['input_ids'].shape[0]}")
        
        print("✓ Data loaders tested successfully!")
        
    except Exception as e:
        print(f"Data loading test failed: {e}")
        print("This is expected if datasets are not yet processed.")
