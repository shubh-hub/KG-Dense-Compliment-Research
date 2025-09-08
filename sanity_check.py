#!/usr/bin/env python3
"""
Sanity check script to validate code before full training runs
Tests all critical components with minimal data to catch errors early
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
from math import comb

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing imports...")
    try:
        from models.baseline_architectures import (
            DenseVectorEncoder, 
            KnowledgeGraphEncoder,
            KGDenseFusionModel,
            get_model_config as get_baseline_config
        )
        from run_complementarity_experiments import (
            create_synthetic_dataset_v2,
            compute_optimal_threshold,
            evaluate_model,
            train_model,
            get_device
        )
        print("  ✅ All imports successful")
        return True
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_model_creation():
    """Test model creation with m1_optimized config"""
    print("🔍 Testing model creation...")
    try:
        from models.baseline_architectures import get_model_config, KGDenseFusionModel
        
        config = get_model_config('m1_optimized')
        print(f"  Config: {config}")
        
        # Test fusion model creation
        model = KGDenseFusionModel(config)
        print(f"  ✅ Fusion model created: {sum(p.numel() for p in model.parameters()):,} params")
        
        # Test dense-only wrapper
        class DenseOnlyModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                from models.baseline_architectures import DenseVectorEncoder
                self.encoder = DenseVectorEncoder(
                    model_name=config.get('dense_model', 'distilbert-base-uncased'),
                    hidden_dim=config.get('hidden_dim', 384),
                    freeze_lm=config.get('freeze_lm', False),
                    pooling=config.get('pooling', 'cls')
                )
                self.qa_head = torch.nn.Linear(config.get('hidden_dim', 384), 1)
            
            def forward(self, batch, task='qa', return_aux_outputs=False):
                embeddings = self.encoder(batch['input_ids'], batch['attention_mask'])
                logits = self.qa_head(embeddings)
                return logits
        
        dense_model = DenseOnlyModel(config)
        print(f"  ✅ Dense model created: {sum(p.numel() for p in dense_model.parameters()):,} params")
        
        return True
    except Exception as e:
        print(f"  ❌ Model creation error: {e}")
        return False

def test_data_generation():
    """Test synthetic data generation"""
    print("🔍 Testing data generation...")
    try:
        from run_complementarity_experiments import create_synthetic_dataset_v2
        
        # Small test dataset
        train_dl, val_dl, test_dl, edge_index, edge_type, meta = create_synthetic_dataset_v2(
            num_samples=40, seq_len=32, num_entities=10, num_relations=5, 
            batch_size=8, split_data=True, seed=42
        )
        
        print(f"  ✅ Dataset created: train={meta['train_size']}, val={meta['val_size']}, test={meta['test_size']}")
        print(f"  Graph: {edge_index.size(1)} edges, {edge_type.max().item() + 1} relations")
        
        # Test batch structure
        for batch in train_dl:
            input_ids, attention_mask, entity_ids, labels, slice_labels = batch
            print(f"  Batch shapes: input_ids={input_ids.shape}, entities={entity_ids.shape}")
            print(f"  Slice distribution: {torch.bincount(slice_labels)}")
            break
            
        return True, (train_dl, val_dl, test_dl, edge_index, edge_type, meta)
    except Exception as e:
        print(f"  ❌ Data generation error: {e}")
        return False, None

def test_forward_pass():
    """Test forward pass with dummy data"""
    print("🔍 Testing forward pass...")
    try:
        from models.baseline_architectures import get_model_config, KGDenseFusionModel
        
        config = get_model_config('m1_optimized')
        model = KGDenseFusionModel(config)
        
        # Create dummy batch
        batch_size = 4
        seq_len = 32
        num_entities = 10
        
        batch = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len),
            'entity_ids': torch.randint(0, num_entities, (batch_size,)),
            'edge_index': torch.randint(0, num_entities, (2, 20)),
            'edge_type': torch.randint(0, 5, (20,))
        }
        
        # Test forward pass
        with torch.no_grad():
            output = model(batch, task='qa')
            print(f"  ✅ Forward pass successful: output shape {output.shape}")
            
            # Test hierarchical gating with aux outputs
            if hasattr(model, 'fusion_type') and model.fusion_type == 'hierarchical_gating':
                output, aux = model(batch, task='qa', return_aux_outputs=True)
                print(f"  ✅ Hierarchical gating aux outputs: {list(aux.keys())}")
        
        return True
    except Exception as e:
        print(f"  ❌ Forward pass error: {e}")
        return False

def test_training_step():
    """Test single training step"""
    print("🔍 Testing training step...")
    try:
        from models.baseline_architectures import get_model_config, KGDenseFusionModel
        from run_complementarity_experiments import get_device
        
        device = get_device()
        config = get_model_config('m1_optimized')
        model = KGDenseFusionModel(config).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Create dummy batch
        batch_size = 4
        seq_len = 32
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones(batch_size, seq_len).to(device)
        entity_ids = torch.randint(0, 10, (batch_size,)).to(device)
        labels = torch.rand(batch_size).to(device)
        slice_labels = torch.randint(0, 4, (batch_size,)).to(device)
        
        edge_index = torch.randint(0, 10, (2, 20)).to(device)
        edge_type = torch.randint(0, 5, (20,)).to(device)
        
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'entity_ids': entity_ids,
            'edge_index': edge_index,
            'edge_type': edge_type
        }
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(batch, task='qa')
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        print(f"  ✅ Training step successful: loss={loss.item():.4f}")
        return True
    except Exception as e:
        print(f"  ❌ Training step error: {e}")
        return False

def test_evaluation_functions():
    """Test evaluation and analysis functions"""
    print("🔍 Testing evaluation functions...")
    try:
        # Test numpy array operations for agreement analysis
        d_preds = np.array([1, 0, 1, 0, 1])
        k_preds = np.array([1, 1, 0, 0, 1])
        agreement = np.mean(d_preds == k_preds)
        print(f"  ✅ Agreement calculation: {agreement:.3f}")
        
        # Test comb function
        result = comb(5, 2)
        print(f"  ✅ Comb function: C(5,2) = {result}")
        
        # Test complementarity analysis components
        labels = np.array([1, 0, 1, 0, 1])
        d_correct = (d_preds == labels)
        k_correct = (k_preds == labels)
        oracle_correct = d_correct | k_correct
        oracle_acc = oracle_correct.mean()
        
        print(f"  ✅ Oracle accuracy: {oracle_acc:.3f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Evaluation function error: {e}")
        return False

def main():
    """Run all sanity checks"""
    print("🧪 Running Sanity Checks Before Full Training")
    print("=" * 60)
    
    checks = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Data Generation", lambda: test_data_generation()[0]),
        ("Forward Pass", test_forward_pass),
        ("Training Step", test_training_step),
        ("Evaluation Functions", test_evaluation_functions)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, test_func in checks:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {name} failed")
        except Exception as e:
            print(f"❌ {name} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Sanity Check Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All checks passed! Safe to run full training.")
        return True
    else:
        print("❌ Some checks failed. Fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
