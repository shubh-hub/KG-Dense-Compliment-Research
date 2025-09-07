#!/usr/bin/env python3
"""
Experiment Tracking and Logging System for KG + Dense Vector Research
Lightweight tracking system optimized for M1 MacBook and free-tier resources
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class ExperimentTracker:
    """Lightweight experiment tracking system"""
    
    def __init__(self, experiment_name: str, output_dir: str = "results/experiments"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Initialize tracking data
        self.config = {}
        self.metrics = defaultdict(list)
        self.logs = []
        self.start_time = time.time()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        self.logger.info(f"Started experiment: {experiment_name}")
        self.logger.info(f"Output directory: {self.experiment_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup experiment logging"""
        logger = logging.getLogger(f"experiment_{self.experiment_name}")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        log_file = self.experiment_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration"""
        self.config.update(config)
        
        # Save config to file
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        self.logger.info("Configuration logged:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_metric(self, name: str, value: Union[float, int], step: Optional[int] = None):
        """Log a metric value"""
        timestamp = time.time() - self.start_time
        
        metric_entry = {
            'value': float(value),
            'timestamp': timestamp,
            'step': step or len(self.metrics[name])
        }
        
        self.metrics[name].append(metric_entry)
        self.logger.info(f"Metric {name}: {value} (step {metric_entry['step']})")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log multiple metrics at once"""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_message(self, message: str, level: str = "info"):
        """Log a custom message"""
        log_entry = {
            'message': message,
            'timestamp': time.time() - self.start_time,
            'level': level
        }
        
        self.logs.append(log_entry)
        
        if level.lower() == "error":
            self.logger.error(message)
        elif level.lower() == "warning":
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def log_model_info(self, model: torch.nn.Module, model_name: str = "model"):
        """Log model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            'model_name': model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }
        
        self.log_config({f"{model_name}_info": model_info})
        
        self.logger.info(f"Model {model_name}:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Model size: {model_info['model_size_mb']:.2f} MB")
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information"""
        self.log_config({'dataset_info': dataset_info})
        
        self.logger.info("Dataset information:")
        for key, value in dataset_info.items():
            self.logger.info(f"  {key}: {value}")
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint_dir = self.experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        checkpoint_file = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_file)
        
        # Save best model if this is the best so far
        if 'best_metric' not in self.config:
            self.config['best_metric'] = metrics.get('f1', metrics.get('map', 0))
            self.config['best_epoch'] = epoch
            best_file = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_file)
        else:
            current_metric = metrics.get('f1', metrics.get('map', 0))
            if current_metric > self.config['best_metric']:
                self.config['best_metric'] = current_metric
                self.config['best_epoch'] = epoch
                best_file = checkpoint_dir / "best_model.pt"
                torch.save(checkpoint, best_file)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    def plot_metrics(self, metric_names: Optional[List[str]] = None):
        """Plot training metrics"""
        if not self.metrics:
            self.logger.warning("No metrics to plot")
            return
        
        if metric_names is None:
            metric_names = list(self.metrics.keys())
        
        # Filter available metrics
        available_metrics = [name for name in metric_names if name in self.metrics]
        
        if not available_metrics:
            self.logger.warning("No available metrics to plot")
            return
        
        # Create plots
        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(10, 4 * len(available_metrics)))
        if len(available_metrics) == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(available_metrics):
            metric_data = self.metrics[metric_name]
            steps = [entry['step'] for entry in metric_data]
            values = [entry['value'] for entry in metric_data]
            
            axes[i].plot(steps, values, marker='o', linewidth=2, markersize=4)
            axes[i].set_title(f'{metric_name.title()} Over Time')
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel(metric_name.title())
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.experiment_dir / "metrics_plot.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Metrics plot saved: {plot_file}")
    
    def generate_report(self) -> str:
        """Generate experiment report"""
        duration = time.time() - self.start_time
        
        report_lines = [
            "=" * 60,
            f"EXPERIMENT REPORT: {self.experiment_name}",
            "=" * 60,
            f"Start Time: {datetime.fromtimestamp(self.start_time)}",
            f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)",
            "",
            "CONFIGURATION:",
            "-" * 20
        ]
        
        for key, value in self.config.items():
            if isinstance(value, dict):
                report_lines.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    report_lines.append(f"  {sub_key}: {sub_value}")
            else:
                report_lines.append(f"{key}: {value}")
        
        report_lines.extend([
            "",
            "FINAL METRICS:",
            "-" * 15
        ])
        
        for metric_name, metric_data in self.metrics.items():
            if metric_data:
                final_value = metric_data[-1]['value']
                best_value = max(entry['value'] for entry in metric_data)
                report_lines.append(f"{metric_name}:")
                report_lines.append(f"  Final: {final_value:.4f}")
                report_lines.append(f"  Best: {best_value:.4f}")
        
        if 'best_epoch' in self.config:
            report_lines.extend([
                "",
                f"Best model saved at epoch {self.config['best_epoch']} with metric {self.config['best_metric']:.4f}"
            ])
        
        report_lines.append("=" * 60)
        
        report_content = '\n'.join(report_lines)
        
        # Save report
        report_file = self.experiment_dir / "experiment_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Experiment report saved: {report_file}")
        return report_content
    
    def save_results(self):
        """Save all experiment results"""
        # Save metrics
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2, default=str)
        
        # Save logs
        logs_file = self.experiment_dir / "logs.json"
        with open(logs_file, 'w') as f:
            json.dump(self.logs, f, indent=2, default=str)
        
        # Generate and save report
        self.generate_report()
        
        # Plot metrics
        self.plot_metrics()
        
        self.logger.info("All experiment results saved")
    
    def finish(self):
        """Finish experiment and save all results"""
        self.log_message("Experiment finished")
        self.save_results()
        
        duration = time.time() - self.start_time
        self.logger.info(f"Total experiment duration: {duration:.2f} seconds")

class ComplementarityTracker:
    """Specialized tracker for complementarity analysis"""
    
    def __init__(self, experiment_tracker: ExperimentTracker):
        self.tracker = experiment_tracker
        self.complementarity_data = defaultdict(list)
    
    def log_model_predictions(self, kg_preds: torch.Tensor, dense_preds: torch.Tensor,
                             fusion_preds: torch.Tensor, labels: torch.Tensor, 
                             step: int, task: str = "qa"):
        """Log predictions from all models for complementarity analysis"""
        # Convert to numpy
        kg_preds = kg_preds.detach().cpu().numpy()
        dense_preds = dense_preds.detach().cpu().numpy()
        fusion_preds = fusion_preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        # Calculate accuracies
        kg_acc = np.mean((kg_preds > 0.5) == labels)
        dense_acc = np.mean((dense_preds > 0.5) == labels)
        fusion_acc = np.mean((fusion_preds > 0.5) == labels)
        
        # Log individual accuracies
        self.tracker.log_metric(f"{task}_kg_accuracy", kg_acc, step)
        self.tracker.log_metric(f"{task}_dense_accuracy", dense_acc, step)
        self.tracker.log_metric(f"{task}_fusion_accuracy", fusion_acc, step)
        
        # Calculate complementarity metrics
        kg_correct = (kg_preds > 0.5) == labels
        dense_correct = (dense_preds > 0.5) == labels
        
        kg_only_correct = kg_correct & ~dense_correct
        dense_only_correct = dense_correct & ~kg_correct
        both_correct = kg_correct & dense_correct
        
        complementarity_score = np.mean(kg_only_correct | dense_only_correct)
        oracle_accuracy = np.mean(kg_correct | dense_correct)
        
        # Log complementarity metrics
        self.tracker.log_metric(f"{task}_complementarity_score", complementarity_score, step)
        self.tracker.log_metric(f"{task}_oracle_accuracy", oracle_accuracy, step)
        self.tracker.log_metric(f"{task}_fusion_improvement", fusion_acc - max(kg_acc, dense_acc), step)
        
        # Store detailed data
        self.complementarity_data[f"{task}_step_{step}"] = {
            'kg_predictions': kg_preds.tolist(),
            'dense_predictions': dense_preds.tolist(),
            'fusion_predictions': fusion_preds.tolist(),
            'labels': labels.tolist(),
            'metrics': {
                'kg_accuracy': float(kg_acc),
                'dense_accuracy': float(dense_acc),
                'fusion_accuracy': float(fusion_acc),
                'complementarity_score': float(complementarity_score),
                'oracle_accuracy': float(oracle_accuracy)
            }
        }
    
    def save_complementarity_analysis(self):
        """Save detailed complementarity analysis"""
        analysis_file = self.tracker.experiment_dir / "complementarity_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(dict(self.complementarity_data), f, indent=2)
        
        self.tracker.logger.info(f"Complementarity analysis saved: {analysis_file}")

class ResourceMonitor:
    """Monitor system resources during experiments"""
    
    def __init__(self, experiment_tracker: ExperimentTracker):
        self.tracker = experiment_tracker
        self.monitoring = False
        
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.tracker.log_message("Started resource monitoring")
        
        # Log initial system info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.tracker.log_config({
                'gpu_name': gpu_name,
                'gpu_memory_gb': gpu_memory
            })
        elif torch.backends.mps.is_available():
            self.tracker.log_config({'device': 'Apple M1 MPS'})
        else:
            self.tracker.log_config({'device': 'CPU'})
    
    def log_memory_usage(self, step: int):
        """Log current memory usage"""
        if not self.monitoring:
            return
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_memory_cached = torch.cuda.memory_reserved(0) / (1024**3)
            
            self.tracker.log_metric("gpu_memory_used_gb", gpu_memory_used, step)
            self.tracker.log_metric("gpu_memory_cached_gb", gpu_memory_cached, step)
        
        # Could add CPU memory monitoring here if needed
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        self.tracker.log_message("Stopped resource monitoring")

def create_experiment_tracker(experiment_name: str, config: Dict[str, Any]) -> ExperimentTracker:
    """Create and configure experiment tracker"""
    tracker = ExperimentTracker(experiment_name)
    tracker.log_config(config)
    
    return tracker

if __name__ == "__main__":
    # Test experiment tracking
    print("Testing experiment tracking...")
    
    # Create test tracker
    tracker = create_experiment_tracker("test_experiment", {
        'model_type': 'fusion',
        'batch_size': 16,
        'learning_rate': 0.001
    })
    
    # Log some test metrics
    for step in range(5):
        tracker.log_metric("loss", 1.0 - step * 0.1, step)
        tracker.log_metric("accuracy", 0.5 + step * 0.1, step)
    
    # Test complementarity tracking
    comp_tracker = ComplementarityTracker(tracker)
    
    # Simulate some predictions
    kg_preds = torch.tensor([0.6, 0.4, 0.8, 0.3])
    dense_preds = torch.tensor([0.7, 0.6, 0.2, 0.9])
    fusion_preds = torch.tensor([0.8, 0.5, 0.7, 0.6])
    labels = torch.tensor([1, 0, 1, 1])
    
    comp_tracker.log_model_predictions(kg_preds, dense_preds, fusion_preds, labels, 0)
    
    # Finish experiment
    tracker.finish()
    
    print("✓ Experiment tracking tested successfully!")
    print(f"Results saved to: {tracker.experiment_dir}")
