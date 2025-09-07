#!/usr/bin/env python3
"""
Quick baseline test on M1 MacBook Air to validate performance
Tests Dense-only, KG-only, and Cross-Attention fusion models
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

def test_model_performance(model, model_name, batch, num_iterations=10):
    """Test model performance and memory usage"""
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            if model_name == 'fusion':
                _ = model(batch, task='qa')
            else:
                _ = model(batch, task='qa')
    
    # Measure performance
    if device.type == 'mps':
        torch.mps.empty_cache()
    
    start_time = time.time()
    memory_before = torch.mps.current_allocated_memory() if device.type == 'mps' else 0
    
    with torch.no_grad():
        for i in range(num_iterations):
            outputs = model(batch, task='qa')
            
            # Simulate loss calculation
            loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), batch['labels'])
    
    end_time = time.time()
    memory_after = torch.mps.current_allocated_memory() if device.type == 'mps' else 0
    
    avg_time = (end_time - start_time) / num_iterations
    memory_used = (memory_after - memory_before) / (1024**2)  # MB
    
    return {
        'avg_inference_time': avg_time,
        'memory_used_mb': memory_used,
        'throughput_samples_per_sec': batch['input_ids'].size(0) / avg_time
    }

def main():
    """Run baseline M1 performance test"""
    print("🚀 Starting M1 MacBook Air Baseline Performance Test")
    print(f"Device: {get_device()}")
    print("-" * 60)
    
    # Test configurations
    batch_sizes = [4, 8, 16]
    model_configs = {
        'lightweight': {'hidden_dim': 384, 'num_entities': 100, 'num_relations': 20}
    }
    
    results = {}
    
    for config_name, config in model_configs.items():
        print(f"\n📊 Testing {config_name.upper()} configuration")
        print(f"Hidden dim: {config['hidden_dim']}, Entities: {config['num_entities']}")
        
        results[config_name] = {}
        
        for batch_size in batch_sizes:
            print(f"\n  Batch size: {batch_size}")
            
            # Create sample batch
            batch = create_sample_batch(
                batch_size=batch_size, 
                num_entities=config['num_entities']
            )
            
            batch_results = {}
            
            # Test models
            model_types = ['dense', 'kg', 'fusion']
            
            for model_type in model_types:
                print(f"    Testing {model_type} model...")
                
                try:
                    # Create model
                    model_config = get_model_config('lightweight')
                    model_config.update(config)
                    
                    if model_type == 'dense':
                        model = BaselineModels.create_dense_only_model(model_config)
                    elif model_type == 'kg':
                        model = BaselineModels.create_kg_only_model(model_config)
                    else:  # fusion
                        model = BaselineModels.create_fusion_model(model_config)
                    
                    # Test inference performance
                    perf_results = test_model_performance(model, model_type, batch)
                    
                    batch_results[model_type] = {
                        **perf_results,
                        'parameters': sum(p.numel() for p in model.parameters()),
                        'status': 'success'
                    }
                    
                    print(f"      ✅ {model_type}: {perf_results['avg_inference_time']:.3f}s, "
                          f"{perf_results['memory_used_mb']:.1f}MB")
                    
                    # Clean up
                    del model
                    if get_device().type == 'mps':
                        torch.mps.empty_cache()
                    
                except Exception as e:
                    print(f"      ❌ {model_type}: Failed - {str(e)}")
                    batch_results[model_type] = {'status': 'failed', 'error': str(e)}
            
            results[config_name][f'batch_{batch_size}'] = batch_results
    
    # Save results
    results_file = Path('results/experiments/m1_baseline_test.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("📈 M1 PERFORMANCE SUMMARY")
    print("="*60)
    
    for config_name, config_results in results.items():
        print(f"\n{config_name.upper()} Configuration:")
        
        for batch_key, batch_results in config_results.items():
            batch_size = batch_key.split('_')[1]
            print(f"  Batch Size {batch_size}:")
            
            for model_type, model_results in batch_results.items():
                if model_results.get('status') == 'success':
                    throughput = model_results['throughput_samples_per_sec']
                    memory = model_results['memory_used_mb']
                    params = model_results['parameters'] / 1e6  # Millions
                    
                    print(f"    {model_type:8}: {throughput:6.1f} samples/sec, "
                          f"{memory:5.1f}MB, {params:4.1f}M params")
                else:
                    print(f"    {model_type:8}: FAILED")
    
    # Performance recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    # Find best performing configuration
    best_config = None
    best_throughput = 0
    
    for config_name, config_results in results.items():
        for batch_key, batch_results in config_results.items():
            fusion_results = batch_results.get('fusion', {})
            if fusion_results.get('status') == 'success':
                throughput = fusion_results['throughput_samples_per_sec']
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_config = (config_name, batch_key.split('_')[1])
    
    if best_config:
        print(f"   • Best performance: {best_config[0]} config with batch size {best_config[1]}")
        print(f"   • Fusion model throughput: {best_throughput:.1f} samples/sec")
        
        if best_throughput > 10:
            print("   • ✅ M1 performance is sufficient for research experiments")
            print("   • 🎯 Recommend continuing development on M1 MacBook Air")
        else:
            print("   • ⚠️  Consider migrating to cloud GPU for larger experiments")
    
    print(f"\n📁 Results saved to: {results_file}")
    print("🏁 M1 baseline test completed!")

if __name__ == "__main__":
    main()
