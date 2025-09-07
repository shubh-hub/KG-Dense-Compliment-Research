#!/usr/bin/env python3
"""
Test all 4 fusion architectures on M1 MacBook Air
Tests Cross-Attention, Hierarchical Gating, Tensor Bilinear, and Contrastive fusion
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.baseline_architectures import BaselineModels, get_model_config

# Set M1 optimizations
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

def get_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def create_sample_batch(batch_size=8, seq_len=128, num_entities=100):
    """Create a sample batch for testing"""
    device = get_device()
    
    batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)).to(device),
        'attention_mask': torch.ones(batch_size, seq_len).to(device),
        'entity_ids': torch.randint(0, num_entities, (batch_size,)).to(device),
        'labels': torch.randint(0, 2, (batch_size,)).float().to(device)
    }
    
    # Create simple edge index for KG (star graph)
    num_edges = batch_size * 5  # 5 edges per entity
    edge_index = torch.randint(0, num_entities, (2, num_edges)).to(device)
    batch['edge_index'] = edge_index
    
    return batch

def test_fusion_architecture(fusion_type, config, batch, num_iterations=10):
    """Test a specific fusion architecture"""
    device = get_device()
    
    # Update config with fusion type
    model_config = config.copy()
    model_config['fusion_type'] = fusion_type
    
    try:
        # Create fusion model
        model = BaselineModels.create_fusion_model(model_config)
        model = model.to(device)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(batch, task='qa')
        
        # Measure performance
        if device.type == 'mps':
            torch.mps.empty_cache()
        
        start_time = time.time()
        memory_before = torch.mps.current_allocated_memory() if device.type == 'mps' else 0
        
        with torch.no_grad():
            for i in range(num_iterations):
                outputs = model(batch, task='qa')
                loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), batch['labels'])
        
        end_time = time.time()
        memory_after = torch.mps.current_allocated_memory() if device.type == 'mps' else 0
        
        avg_time = (end_time - start_time) / num_iterations
        memory_used = (memory_after - memory_before) / (1024**2)  # MB
        
        results = {
            'avg_inference_time': avg_time,
            'memory_used_mb': memory_used,
            'throughput_samples_per_sec': batch['input_ids'].size(0) / avg_time,
            'parameters': sum(p.numel() for p in model.parameters()),
            'status': 'success'
        }
        
        # Clean up
        del model
        if device.type == 'mps':
            torch.mps.empty_cache()
        
        return results
        
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

def main():
    """Test all fusion architectures"""
    print("🚀 Testing All 4 Fusion Architectures on M1 MacBook Air")
    print(f"Device: {get_device()}")
    print("=" * 70)
    
    # Test configuration
    batch_size = 8
    config = get_model_config('lightweight')
    config.update({
        'num_entities': 100,
        'num_relations': 20,
        'hidden_dim': 384
    })
    
    # Create sample batch
    batch = create_sample_batch(batch_size=batch_size, num_entities=config['num_entities'])
    
    # Test all fusion architectures
    fusion_types = ['cross_attention', 'hierarchical_gating', 'tensor_bilinear', 'contrastive']
    results = {}
    
    for fusion_type in fusion_types:
        print(f"\n🔬 Testing {fusion_type.replace('_', ' ').title()} Fusion...")
        
        result = test_fusion_architecture(fusion_type, config, batch)
        results[fusion_type] = result
        
        if result['status'] == 'success':
            throughput = result['throughput_samples_per_sec']
            memory = result['memory_used_mb']
            params = result['parameters'] / 1e6  # Millions
            
            print(f"   ✅ Success: {throughput:6.1f} samples/sec, {memory:5.1f}MB, {params:4.1f}M params")
        else:
            print(f"   ❌ Failed: {result['error']}")
    
    # Save results
    results_file = Path('results/experiments/fusion_architectures_test.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 FUSION ARCHITECTURES PERFORMANCE SUMMARY")
    print("=" * 70)
    
    successful_tests = {k: v for k, v in results.items() if v.get('status') == 'success'}
    
    if successful_tests:
        print(f"\n✅ Successfully tested {len(successful_tests)}/4 fusion architectures:")
        
        # Sort by throughput
        sorted_results = sorted(successful_tests.items(), 
                              key=lambda x: x[1]['throughput_samples_per_sec'], 
                              reverse=True)
        
        for i, (fusion_type, result) in enumerate(sorted_results, 1):
            throughput = result['throughput_samples_per_sec']
            memory = result['memory_used_mb']
            params = result['parameters'] / 1e6
            
            print(f"  {i}. {fusion_type.replace('_', ' ').title():20}: "
                  f"{throughput:6.1f} samples/sec, {memory:5.1f}MB, {params:4.1f}M params")
        
        # Best performing architecture
        best_fusion = sorted_results[0]
        best_throughput = best_fusion[1]['throughput_samples_per_sec']
        
        print(f"\n🏆 Best performing: {best_fusion[0].replace('_', ' ').title()}")
        print(f"   Throughput: {best_throughput:.1f} samples/sec")
        
        if best_throughput > 50:
            print("   🎯 All fusion architectures are ready for complementarity experiments!")
        else:
            print("   ⚠️  Consider optimizing for larger-scale experiments")
    
    else:
        print("❌ No fusion architectures passed the test")
    
    print(f"\n📁 Results saved to: {results_file}")
    print("🏁 Fusion architecture testing completed!")

if __name__ == "__main__":
    main()
