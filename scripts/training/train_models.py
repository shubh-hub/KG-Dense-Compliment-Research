#!/usr/bin/env python3
"""
Training Pipeline for KG + Dense Vector Complementarity Research
Optimized for Apple M1 MacBook with comprehensive experiment tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.baseline_architectures import (
    create_dense_model, create_kg_model, create_fusion_model
)
from data_processing.data_loaders import create_data_processor, DataConfig
from evaluation.evaluation_framework import QAEvaluator, IREvaluator, ComplementarityAnalyzer
from utils.experiment_tracker import (
    create_experiment_tracker, ComplementarityTracker, ResourceMonitor
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Main training class for KG + Dense Vector models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._get_device()
        
        # Initialize experiment tracking
        self.experiment_tracker = create_experiment_tracker(
            experiment_name=config['experiment_name'],
            config=config
        )
        self.complementarity_tracker = ComplementarityTracker(self.experiment_tracker)
        self.resource_monitor = ResourceMonitor(self.experiment_tracker)
        
        # Initialize data processor
        data_config = DataConfig.lightweight_config() if config.get('lightweight', True) else DataConfig.standard_config()
        self.data_processor = create_data_processor(data_config)
        
        # Initialize models
        self.models = self._initialize_models()
        self.optimizers = self._initialize_optimizers()
        self.schedulers = self._initialize_schedulers()
        
        # Initialize evaluators
        self.qa_evaluator = QAEvaluator()
        self.ir_evaluator = IREvaluator()
        self.complementarity_analyzer = ComplementarityAnalyzer()
        
        logger.info(f"Training initialized on device: {self.device}")
        self.experiment_tracker.log_config({'device': str(self.device)})
    
    def _get_device(self) -> torch.device:
        """Get the best available device"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _initialize_models(self) -> Dict[str, nn.Module]:
        """Initialize all models"""
        models = {}
        
        # Model configurations
        model_size = self.config.get('model_size', 'lightweight')
        
        # Dense model
        models['dense'] = create_dense_model(
            model_type='dense_only',
            size=model_size,
            num_classes=self.config.get('num_classes', 2)
        ).to(self.device)
        
        # KG model
        models['kg'] = create_kg_model(
            model_type='kg_only',
            size=model_size,
            num_entities=self.config.get('num_entities', 1000),
            num_relations=self.config.get('num_relations', 100),
            num_classes=self.config.get('num_classes', 2)
        ).to(self.device)
        
        # Fusion model
        models['fusion'] = create_fusion_model(
            model_type='fusion',
            size=model_size,
            fusion_method=self.config.get('fusion_method', 'cross_attention'),
            num_entities=self.config.get('num_entities', 1000),
            num_relations=self.config.get('num_relations', 100),
            num_classes=self.config.get('num_classes', 2)
        ).to(self.device)
        
        # Log model information
        for name, model in models.items():
            self.experiment_tracker.log_model_info(model, name)
        
        return models
    
    def _initialize_optimizers(self) -> Dict[str, optim.Optimizer]:
        """Initialize optimizers for all models"""
        optimizers = {}
        lr = self.config.get('learning_rate', 1e-4)
        
        for name, model in self.models.items():
            optimizers[name] = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
        
        return optimizers
    
    def _initialize_schedulers(self) -> Dict[str, Any]:
        """Initialize learning rate schedulers"""
        schedulers = {}
        
        for name, optimizer in self.optimizers.items():
            schedulers[name] = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=3,
                verbose=True
            )
        
        return schedulers
    
    def train_epoch(self, dataloader: DataLoader, epoch: int, task: str = 'qa') -> Dict[str, float]:
        """Train all models for one epoch"""
        # Set models to training mode
        for model in self.models.values():
            model.train()
        
        epoch_losses = {name: 0.0 for name in self.models.keys()}
        epoch_accuracies = {name: 0.0 for name in self.models.keys()}
        num_batches = 0
        
        # Loss function
        criterion = nn.BCEWithLogitsLoss() if self.config.get('num_classes', 2) == 2 else nn.CrossEntropyLoss()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} ({task})")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            batch_losses = {}
            batch_predictions = {}
            
            # Train each model
            for model_name, model in self.models.items():
                optimizer = self.optimizers[model_name]
                optimizer.zero_grad()
                
                # Forward pass
                if model_name == 'dense':
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                elif model_name == 'kg':
                    outputs = model(
                        entity_ids=batch['entity_ids'],
                        edge_index=batch.get('edge_index'),
                        edge_type=batch.get('edge_type')
                    )
                else:  # fusion
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        entity_ids=batch['entity_ids'],
                        edge_index=batch.get('edge_index'),
                        edge_type=batch.get('edge_type')
                    )
                
                # Calculate loss
                if task == 'qa':
                    labels = batch['labels']
                else:  # ir
                    labels = batch['relevance']
                
                if self.config.get('num_classes', 2) == 2:
                    loss = criterion(outputs.squeeze(), labels.float())
                    predictions = torch.sigmoid(outputs.squeeze())
                else:
                    loss = criterion(outputs, labels.long())
                    predictions = torch.softmax(outputs, dim=1)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Store results
                batch_losses[model_name] = loss.item()
                batch_predictions[model_name] = predictions.detach()
                
                # Update running averages
                epoch_losses[model_name] += loss.item()
                
                # Calculate accuracy
                if self.config.get('num_classes', 2) == 2:
                    correct = ((predictions > 0.5) == labels).float().mean()
                else:
                    correct = (predictions.argmax(dim=1) == labels).float().mean()
                
                epoch_accuracies[model_name] += correct.item()
            
            # Log complementarity metrics
            if len(batch_predictions) >= 3:  # All models available
                self.complementarity_tracker.log_model_predictions(
                    kg_preds=batch_predictions['kg'],
                    dense_preds=batch_predictions['dense'],
                    fusion_preds=batch_predictions['fusion'],
                    labels=labels,
                    step=epoch * len(dataloader) + batch_idx,
                    task=task
                )
            
            num_batches += 1
            
            # Update progress bar
            avg_loss = np.mean(list(batch_losses.values()))
            progress_bar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
            
            # Log memory usage periodically
            if batch_idx % 10 == 0:
                self.resource_monitor.log_memory_usage(epoch * len(dataloader) + batch_idx)
        
        # Calculate epoch averages
        epoch_metrics = {}
        for model_name in self.models.keys():
            epoch_metrics[f'{task}_{model_name}_loss'] = epoch_losses[model_name] / num_batches
            epoch_metrics[f'{task}_{model_name}_accuracy'] = epoch_accuracies[model_name] / num_batches
        
        return epoch_metrics
    
    def evaluate_epoch(self, dataloader: DataLoader, epoch: int, task: str = 'qa') -> Dict[str, float]:
        """Evaluate all models for one epoch"""
        # Set models to evaluation mode
        for model in self.models.values():
            model.eval()
        
        all_predictions = {name: [] for name in self.models.keys()}
        all_labels = []
        all_ids = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch} ({task})"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get predictions from each model
                for model_name, model in self.models.items():
                    if model_name == 'dense':
                        outputs = model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask']
                        )
                    elif model_name == 'kg':
                        outputs = model(
                            entity_ids=batch['entity_ids'],
                            edge_index=batch.get('edge_index'),
                            edge_type=batch.get('edge_type')
                        )
                    else:  # fusion
                        outputs = model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            entity_ids=batch['entity_ids'],
                            edge_index=batch.get('edge_index'),
                            edge_type=batch.get('edge_type')
                        )
                    
                    if self.config.get('num_classes', 2) == 2:
                        predictions = torch.sigmoid(outputs.squeeze())
                    else:
                        predictions = torch.softmax(outputs, dim=1)
                    
                    all_predictions[model_name].extend(predictions.cpu().numpy())
                
                # Store labels and IDs
                if task == 'qa':
                    labels = batch['labels']
                    ids = batch.get('example_ids', [])
                else:  # ir
                    labels = batch['relevance']
                    ids = batch.get('query_ids', [])
                
                all_labels.extend(labels.cpu().numpy())
                all_ids.extend(ids)
        
        # Calculate evaluation metrics
        eval_metrics = {}
        
        for model_name, predictions in all_predictions.items():
            predictions = np.array(predictions)
            labels = np.array(all_labels)
            
            if task == 'qa':
                metrics = self.qa_evaluator.evaluate_predictions(predictions, labels)
            else:  # ir
                # For IR, we need query-document pairs
                metrics = self.ir_evaluator.evaluate_predictions(predictions, labels)
            
            # Add model name prefix
            for metric_name, value in metrics.items():
                eval_metrics[f'{task}_{model_name}_{metric_name}'] = value
        
        # Calculate complementarity metrics
        if len(all_predictions) >= 3:
            comp_metrics = self.complementarity_analyzer.analyze_complementarity(
                kg_predictions=np.array(all_predictions['kg']),
                dense_predictions=np.array(all_predictions['dense']),
                fusion_predictions=np.array(all_predictions['fusion']),
                labels=np.array(all_labels)
            )
            
            # Add complementarity metrics
            for metric_name, value in comp_metrics.items():
                eval_metrics[f'{task}_complementarity_{metric_name}'] = value
        
        return eval_metrics
    
    def train(self):
        """Main training loop"""
        self.experiment_tracker.log_message("Starting training...")
        self.resource_monitor.start_monitoring()
        
        try:
            # Create dataloaders
            dataloaders = self.data_processor.create_dataloaders(
                batch_size=self.config.get('batch_size', 8),
                num_workers=self.config.get('num_workers', 1)
            )
            
            # Log dataset information
            dataset_info = {}
            for name, loader in dataloaders.items():
                if hasattr(loader, '__len__'):
                    dataset_info[name] = len(loader)
            
            self.experiment_tracker.log_dataset_info(dataset_info)
            
            # Training loop
            num_epochs = self.config.get('num_epochs', 10)
            best_metrics = {}
            
            for epoch in range(num_epochs):
                self.experiment_tracker.log_message(f"Starting epoch {epoch + 1}/{num_epochs}")
                
                epoch_metrics = {}
                
                # Train on available tasks
                if 'qa_dataloader' in dataloaders:
                    train_metrics = self.train_epoch(dataloaders['qa_dataloader'], epoch, 'qa')
                    epoch_metrics.update(train_metrics)
                    
                    # Evaluate
                    eval_metrics = self.evaluate_epoch(dataloaders['qa_dataloader'], epoch, 'qa')
                    epoch_metrics.update(eval_metrics)
                
                if 'ir_dataloader' in dataloaders:
                    train_metrics = self.train_epoch(dataloaders['ir_dataloader'], epoch, 'ir')
                    epoch_metrics.update(train_metrics)
                    
                    # Evaluate
                    eval_metrics = self.evaluate_epoch(dataloaders['ir_dataloader'], epoch, 'ir')
                    epoch_metrics.update(eval_metrics)
                
                # Log all metrics
                self.experiment_tracker.log_metrics(epoch_metrics, epoch)
                
                # Update learning rate schedulers
                for name, scheduler in self.schedulers.items():
                    # Use best available metric for scheduling
                    metric_key = f'qa_{name}_f1' if f'qa_{name}_f1' in epoch_metrics else f'ir_{name}_map'
                    if metric_key in epoch_metrics:
                        scheduler.step(epoch_metrics[metric_key])
                
                # Save checkpoint
                if epoch_metrics:
                    # Find best model for checkpointing
                    fusion_metrics = {k: v for k, v in epoch_metrics.items() if 'fusion' in k and ('f1' in k or 'map' in k)}
                    if fusion_metrics:
                        best_metric = max(fusion_metrics.values())
                        self.experiment_tracker.save_checkpoint(
                            self.models['fusion'], 
                            self.optimizers['fusion'], 
                            epoch, 
                            {'best_metric': best_metric}
                        )
                
                # Early stopping check
                if self.config.get('early_stopping', False):
                    if self._should_early_stop(epoch_metrics, best_metrics):
                        self.experiment_tracker.log_message("Early stopping triggered")
                        break
                
                best_metrics.update(epoch_metrics)
            
            # Save complementarity analysis
            self.complementarity_tracker.save_complementarity_analysis()
            
        except Exception as e:
            self.experiment_tracker.log_message(f"Training failed: {str(e)}", "error")
            raise
        
        finally:
            self.resource_monitor.stop_monitoring()
            self.experiment_tracker.finish()
    
    def _should_early_stop(self, current_metrics: Dict[str, float], 
                          best_metrics: Dict[str, float]) -> bool:
        """Check if training should stop early"""
        patience = self.config.get('early_stopping_patience', 5)
        
        # Simple early stopping based on fusion model performance
        fusion_metrics = [k for k in current_metrics.keys() if 'fusion' in k and ('f1' in k or 'map' in k)]
        
        if not fusion_metrics:
            return False
        
        current_best = max(current_metrics[k] for k in fusion_metrics)
        historical_best = max(best_metrics.get(k, 0) for k in fusion_metrics)
        
        # This is a simplified early stopping - in practice you'd track patience properly
        return current_best < historical_best * 0.95

def create_training_config(
    experiment_name: str = "kg_dense_complementarity",
    model_size: str = "lightweight",
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    **kwargs
) -> Dict[str, Any]:
    """Create training configuration"""
    
    config = {
        'experiment_name': experiment_name,
        'model_size': model_size,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': 1e-5,
        'num_classes': 2,
        'num_entities': 1000,
        'num_relations': 100,
        'fusion_method': 'cross_attention',
        'lightweight': True,
        'num_workers': 1,
        'early_stopping': True,
        'early_stopping_patience': 5
    }
    
    config.update(kwargs)
    return config

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train KG + Dense Vector models")
    parser.add_argument('--experiment_name', type=str, default='kg_dense_baseline',
                       help='Name of the experiment')
    parser.add_argument('--model_size', type=str, choices=['lightweight', 'standard', 'large'],
                       default='lightweight', help='Model size configuration')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--fusion_method', type=str, 
                       choices=['cross_attention', 'hierarchical_gating'],
                       default='cross_attention', help='Fusion method')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_training_config(
        experiment_name=args.experiment_name,
        model_size=args.model_size,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fusion_method=args.fusion_method
    )
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Start training
    trainer.train()
    
    print(f"Training completed! Results saved to: {trainer.experiment_tracker.experiment_dir}")

if __name__ == "__main__":
    main()
