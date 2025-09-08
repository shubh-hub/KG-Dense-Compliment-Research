#!/usr/bin/env python3

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.baseline_architectures import HierarchicalGatingFusion, KGDenseFusionModel

def debug_hierarchical_gating():
    """Debug the hierarchical gating fusion shape issues"""
    
    # Create test inputs
    batch_size = 4
    hidden_dim = 384
    
    kg_embeddings = torch.randn(batch_size, hidden_dim)
    dense_embeddings = torch.randn(batch_size, hidden_dim)
    
    print("🔍 Testing HierarchicalGatingFusion directly...")
    fusion = HierarchicalGatingFusion(hidden_dim=hidden_dim)
    
    try:
        # Test basic forward
        print("  Testing basic forward...")
        output = fusion(kg_embeddings, dense_embeddings)
        print(f"  ✅ Basic output shape: {output.shape}")
        
        # Test with gates
        print("  Testing with gates...")
        result = fusion(kg_embeddings, dense_embeddings, return_gates=True)
        print(f"  ✅ With gates - outputs: {len(result)}")
        for i, out in enumerate(result):
            print(f"    Output {i}: {out.shape}")
        
        # Test with aux logits
        print("  Testing with aux logits...")
        result = fusion(kg_embeddings, dense_embeddings, return_aux_logits=True)
        print(f"  ✅ With aux logits - outputs: {len(result)}")
        for i, out in enumerate(result):
            print(f"    Output {i}: {out.shape}")
        
        # Test with both
        print("  Testing with both gates and aux logits...")
        result = fusion(kg_embeddings, dense_embeddings, return_gates=True, return_aux_logits=True)
        print(f"  ✅ With both - outputs: {len(result)}")
        for i, out in enumerate(result):
            print(f"    Output {i}: {out.shape}")
            
    except Exception as e:
        print(f"  ❌ HierarchicalGatingFusion failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🔍 Testing KGDenseFusionModel with hierarchical gating...")
    
    # Create test batch
    batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, 32)),
        'attention_mask': torch.ones(batch_size, 32),
        'entity_ids': torch.randint(0, 100, (batch_size,)),
        'edge_index': torch.randint(0, 100, (2, 200)),
        'edge_type': torch.randint(0, 20, (200,))
    }
    
    try:
        config = {
            'dense_model': 'distilbert-base-uncased',
            'hidden_dim': hidden_dim,
            'num_entities': 100,
            'num_relations': 20,
            'fusion_type': 'hierarchical_gating',
            'freeze_lm': True,
            'dropout': 0.1,
            'gnn_type': 'rgcn',
            'gnn_layers': 2,
            'num_bases': 16
        }
        model = KGDenseFusionModel(config)
        
        # Test basic forward
        print("  Testing basic forward...")
        output = model(batch, task='qa')
        print(f"  ✅ Basic output shape: {output.shape}")
        
        # Test with aux outputs
        print("  Testing with aux outputs...")
        result = model(batch, task='qa', return_aux_outputs=True)
        if isinstance(result, tuple):
            output, aux = result
            print(f"  ✅ Output shape: {output.shape}")
            print(f"  ✅ Aux outputs keys: {aux.keys() if aux else 'None'}")
            if aux:
                for key, val in aux.items():
                    print(f"    {key}: {val.shape}")
        else:
            print(f"  ✅ Single output shape: {result.shape}")
            
    except Exception as e:
        print(f"  ❌ KGDenseFusionModel failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_hierarchical_gating()
