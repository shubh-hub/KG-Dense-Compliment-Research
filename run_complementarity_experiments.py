#!/usr/bin/env python3
"""
Baseline Complementarity Training Experiments
Systematic evaluation of Dense-only, KG-only, and all 4 fusion architectures
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import time
from pathlib import Path
import sys
import os
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.baseline_architectures import BaselineModels, get_model_config
from data_processing.data_loaders import create_data_processor, DataConfig
from evaluation.evaluation_framework import QAEvaluator, ComplementarityAnalyzer
from utils.experiment_tracker import create_experiment_tracker

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

def create_synthetic_dataset(num_samples=1000, seq_len=128, num_entities=100, batch_size=16):
    """Create synthetic dataset for initial complementarity testing"""
    device = get_device()
    
    # Generate synthetic data
    input_ids = torch.randint(0, 1000, (num_samples, seq_len))
    attention_mask = torch.ones(num_samples, seq_len)
    entity_ids = torch.randint(0, num_entities, (num_samples,))
    
    # Create edge indices (simple star graphs)
    edge_indices = []
    for i in range(num_samples):
        num_edges = 5  # 5 edges per sample
        edge_index = torch.randint(0, num_entities, (2, num_edges))
        edge_indices.append(edge_index)
    
    # Generate labels with some complementarity patterns
    # Dense features work better for certain patterns, KG for others
    dense_scores = torch.randn(num_samples)
    kg_scores = torch.randn(num_samples)
    
    # Create complementarity: some samples favor dense, others favor KG
    complementarity_mask = torch.rand(num_samples) > 0.5
    labels = torch.where(
        complementarity_mask,
        (dense_scores > 0).float(),  # Dense-favorable samples
        (kg_scores > 0).float()      # KG-favorable samples
    )
    
    # Create dataset
    dataset = TensorDataset(input_ids, attention_mask, entity_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, edge_indices

def train_model(model, model_name, dataloader, edge_indices, num_epochs=5, lr=1e-4):
    """Train a model and return performance metrics"""
    device = get_device()
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    epoch_losses = []
    epoch_accuracies = []
    
    print(f"  Training {model_name} model...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, (input_ids, attention_mask, entity_ids, labels) in enumerate(progress_bar):
            # Move to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            entity_ids = entity_ids.to(device)
            labels = labels.to(device)
            
            # Get corresponding edge indices
            batch_size = input_ids.size(0)
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(edge_indices))
            
            # Create batch edge index
            edge_index = torch.cat([edge_indices[i] for i in range(start_idx, end_idx)], dim=1).to(device)
            
            # Create batch dictionary
            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'entity_ids': entity_ids,
                'edge_index': edge_index
            }
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch, task='qa')
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.sigmoid(outputs.squeeze()) > 0.5
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_predictions/total_samples:.3f}'
            })
        
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = correct_predictions / total_samples
        
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)
        
        print(f"    Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.3f}")
    
    return {
        'final_loss': epoch_losses[-1],
        'final_accuracy': epoch_accuracies[-1],
        'loss_history': epoch_losses,
        'accuracy_history': epoch_accuracies
    }

def evaluate_model(model, model_name, dataloader, edge_indices):
    """Evaluate model and return detailed metrics"""
    device = get_device()
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print(f"  Evaluating {model_name} model...")
    
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, entity_ids, labels) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            # Move to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            entity_ids = entity_ids.to(device)
            labels = labels.to(device)
            
            # Get corresponding edge indices
            batch_size = input_ids.size(0)
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(edge_indices))
            
            # Create batch edge index
            edge_index = torch.cat([edge_indices[i] for i in range(start_idx, end_idx)], dim=1).to(device)
            
            # Create batch dictionary
            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'entity_ids': entity_ids,
                'edge_index': edge_index
            }
            
            # Forward pass
            outputs = model(batch, task='qa')
            probabilities = torch.sigmoid(outputs.squeeze())
            predictions = probabilities > 0.5
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    probabilities = np.array(all_probabilities)
    
    accuracy = (predictions == labels).mean()
    
    # Calculate precision, recall, F1
    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
        'probabilities': probabilities,
        'labels': labels
    }

def run_complementarity_experiment(config_size='lightweight', num_samples=1000, num_epochs=5):
    """Run comprehensive complementarity experiment"""
    print("🚀 Starting Baseline Complementarity Training Experiments")
    print(f"Configuration: {config_size}, Samples: {num_samples}, Epochs: {num_epochs}")
    print("=" * 80)
    
    device = get_device()
    print(f"Device: {device}")
    
    # Create synthetic dataset
    print("\n📊 Creating synthetic dataset...")
    train_dataloader, train_edge_indices = create_synthetic_dataset(
        num_samples=num_samples, batch_size=16
    )
    test_dataloader, test_edge_indices = create_synthetic_dataset(
        num_samples=num_samples//4, batch_size=16
    )
    
    # Get model configuration
    config = get_model_config(config_size)
    config.update({
        'num_entities': 100,
        'num_relations': 20,
        'hidden_dim': 384
    })
    
    # Model types to test
    model_configs = [
        ('dense_only', 'Dense-Only Baseline'),
        ('kg_only', 'KG-Only Baseline'),
        ('fusion_cross_attention', 'Cross-Attention Fusion'),
        ('fusion_hierarchical_gating', 'Hierarchical Gating Fusion'),
        ('fusion_tensor_bilinear', 'Tensor Bilinear Fusion'),
        ('fusion_contrastive', 'Contrastive Fusion')
    ]
    
    results = {}
    trained_models = {}
    
    # Train and evaluate each model
    for model_key, model_name in model_configs:
        print(f"\n🔬 Testing {model_name}")
        print("-" * 50)
        
        try:
            # Create model
            if model_key == 'dense_only':
                model = BaselineModels.create_dense_only_model(config)
            elif model_key == 'kg_only':
                model = BaselineModels.create_kg_only_model(config)
            else:  # fusion models
                fusion_type = model_key.replace('fusion_', '')
                fusion_config = config.copy()
                fusion_config['fusion_type'] = fusion_type
                model = BaselineModels.create_fusion_model(fusion_config)
            
            # Train model
            train_results = train_model(
                model, model_name, train_dataloader, train_edge_indices, 
                num_epochs=num_epochs
            )
            
            # Evaluate model
            eval_results = evaluate_model(
                model, model_name, test_dataloader, test_edge_indices
            )
            
            # Store results
            results[model_key] = {
                'model_name': model_name,
                'train_results': train_results,
                'eval_results': eval_results,
                'parameters': sum(p.numel() for p in model.parameters()),
                'status': 'success'
            }
            
            trained_models[model_key] = model
            
            print(f"  ✅ {model_name}: F1 = {eval_results['f1']:.3f}, Acc = {eval_results['accuracy']:.3f}")
            
        except Exception as e:
            print(f"  ❌ {model_name}: Failed - {str(e)}")
            results[model_key] = {
                'model_name': model_name,
                'status': 'failed',
                'error': str(e)
            }
    
    # Analyze complementarity
    print(f"\n🔍 Analyzing Complementarity...")
    print("-" * 50)
    
    successful_models = {k: v for k, v in results.items() if v.get('status') == 'success'}
    
    if len(successful_models) >= 3:  # Need at least dense, kg, and one fusion
        complementarity_analysis = analyze_complementarity(successful_models)
        results['complementarity_analysis'] = complementarity_analysis
        
        print("Complementarity Analysis:")
        for metric, value in complementarity_analysis.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
    
    # Save results
    results_file = Path('results/experiments/complementarity_baseline_results.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    for model_key, model_results in results.items():
        if model_results.get('status') == 'success' and 'eval_results' in model_results:
            eval_results = model_results['eval_results']
            for key in ['predictions', 'probabilities', 'labels']:
                if key in eval_results and isinstance(eval_results[key], np.ndarray):
                    eval_results[key] = eval_results[key].tolist()
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print final summary
    print_experiment_summary(results)
    print(f"\n📁 Results saved to: {results_file}")
    print("🏁 Complementarity experiments completed!")
    
    return results

def analyze_complementarity(successful_models):
    """Analyze complementarity between models"""
    analysis = {}
    
    # Get model predictions
    dense_results = successful_models.get('dense_only', {}).get('eval_results', {})
    kg_results = successful_models.get('kg_only', {}).get('eval_results', {})
    
    if dense_results and kg_results:
        dense_preds = np.array(dense_results['predictions'])
        kg_preds = np.array(kg_results['predictions'])
        labels = np.array(dense_results['labels'])
        
        # Agreement analysis
        agreement = (dense_preds == kg_preds).mean()
        analysis['dense_kg_agreement'] = agreement
        
        # Complementarity score (when one is right and other is wrong)
        dense_correct = (dense_preds == labels)
        kg_correct = (kg_preds == labels)
        
        complementarity_cases = ((dense_correct & ~kg_correct) | (~dense_correct & kg_correct)).sum()
        total_cases = len(labels)
        analysis['complementarity_ratio'] = complementarity_cases / total_cases
        
        # Individual accuracies
        analysis['dense_accuracy'] = dense_correct.mean()
        analysis['kg_accuracy'] = kg_correct.mean()
        
        # Potential fusion improvement
        oracle_correct = (dense_correct | kg_correct)
        analysis['oracle_accuracy'] = oracle_correct.mean()
        analysis['potential_improvement'] = analysis['oracle_accuracy'] - max(analysis['dense_accuracy'], analysis['kg_accuracy'])
    
    # Fusion model performance
    fusion_models = {k: v for k, v in successful_models.items() if k.startswith('fusion_')}
    if fusion_models:
        fusion_accuracies = [v['eval_results']['accuracy'] for v in fusion_models.values()]
        analysis['best_fusion_accuracy'] = max(fusion_accuracies)
        analysis['avg_fusion_accuracy'] = np.mean(fusion_accuracies)
        
        # Compare with baselines
        baseline_accuracies = []
        if 'dense_only' in successful_models:
            baseline_accuracies.append(successful_models['dense_only']['eval_results']['accuracy'])
        if 'kg_only' in successful_models:
            baseline_accuracies.append(successful_models['kg_only']['eval_results']['accuracy'])
        
        if baseline_accuracies:
            best_baseline = max(baseline_accuracies)
            analysis['fusion_improvement'] = analysis['best_fusion_accuracy'] - best_baseline
    
    return analysis

def print_experiment_summary(results):
    """Print comprehensive experiment summary"""
    print("\n" + "=" * 80)
    print("📊 COMPLEMENTARITY EXPERIMENT SUMMARY")
    print("=" * 80)
    
    successful_models = {k: v for k, v in results.items() if v.get('status') == 'success'}
    
    if successful_models:
        print(f"\n✅ Successfully trained {len(successful_models)} models:")
        
        # Sort by F1 score
        sorted_models = sorted(
            successful_models.items(),
            key=lambda x: x[1]['eval_results']['f1'],
            reverse=True
        )
        
        print("\nModel Performance Ranking:")
        for i, (model_key, model_results) in enumerate(sorted_models, 1):
            eval_results = model_results['eval_results']
            params = model_results['parameters'] / 1e6
            
            print(f"  {i}. {model_results['model_name']:25}: "
                  f"F1={eval_results['f1']:.3f}, "
                  f"Acc={eval_results['accuracy']:.3f}, "
                  f"Params={params:.1f}M")
        
        # Complementarity analysis
        if 'complementarity_analysis' in results:
            comp_analysis = results['complementarity_analysis']
            print(f"\n🔍 Complementarity Analysis:")
            print(f"  Dense-KG Agreement: {comp_analysis.get('dense_kg_agreement', 0):.3f}")
            print(f"  Complementarity Ratio: {comp_analysis.get('complementarity_ratio', 0):.3f}")
            print(f"  Potential Improvement: {comp_analysis.get('potential_improvement', 0):.3f}")
            print(f"  Fusion Improvement: {comp_analysis.get('fusion_improvement', 0):.3f}")
            
            if comp_analysis.get('fusion_improvement', 0) > 0.05:
                print("  🎯 Significant complementarity detected! Fusion models show clear benefits.")
            else:
                print("  ⚠️  Limited complementarity observed. Consider different fusion strategies.")
    
    else:
        print("❌ No models completed successfully")

def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description="Run complementarity experiments")
    parser.add_argument('--config_size', type=str, default='lightweight',
                       choices=['lightweight', 'standard', 'large'],
                       help='Model configuration size')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of training samples')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Run experiments
    results = run_complementarity_experiment(
        config_size=args.config_size,
        num_samples=args.num_samples,
        num_epochs=args.num_epochs
    )
    
    return results

if __name__ == "__main__":
    main()
