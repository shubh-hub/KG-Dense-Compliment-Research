#!/usr/bin/env python3
"""
Evaluation Framework for KG + Dense Vector Complementarity Research
Implements metrics and evaluation protocols for QA and IR tasks
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from collections import defaultdict
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class QAEvaluator:
    """Evaluator for Question Answering tasks"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset evaluation metrics"""
        self.predictions = []
        self.ground_truth = []
        self.scores = []
        
    def add_batch(self, predictions: torch.Tensor, labels: torch.Tensor, 
                  scores: Optional[torch.Tensor] = None):
        """
        Add batch predictions for evaluation
        
        Args:
            predictions: Model predictions [batch_size]
            labels: Ground truth labels [batch_size]
            scores: Confidence scores [batch_size] (optional)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if scores is not None and isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        self.predictions.extend(predictions.flatten())
        self.ground_truth.extend(labels.flatten())
        
        if scores is not None:
            self.scores.extend(scores.flatten())
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute QA evaluation metrics"""
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        ground_truth = np.array(self.ground_truth)
        
        # Convert to binary predictions if needed
        if predictions.dtype == np.float32 or predictions.dtype == np.float64:
            binary_preds = (predictions > 0.5).astype(int)
        else:
            binary_preds = predictions.astype(int)
        
        # Basic metrics
        accuracy = accuracy_score(ground_truth, binary_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, binary_preds, average='binary', zero_division=0
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
        
        # AUC if scores available
        if self.scores:
            try:
                auc = roc_auc_score(ground_truth, self.scores)
                metrics['auc'] = float(auc)
            except ValueError:
                metrics['auc'] = 0.0
        
        return metrics

class IREvaluator:
    """Evaluator for Information Retrieval tasks"""
    
    def __init__(self, k_values: List[int] = [1, 5, 10, 20]):
        self.k_values = k_values
        self.reset()
    
    def reset(self):
        """Reset evaluation metrics"""
        self.query_results = defaultdict(list)  # query_id -> [(doc_id, score, relevance)]
        
    def add_batch(self, query_ids: List[str], doc_ids: List[str], 
                  scores: torch.Tensor, relevance: torch.Tensor):
        """
        Add batch predictions for evaluation
        
        Args:
            query_ids: Query identifiers
            doc_ids: Document identifiers  
            scores: Relevance scores [batch_size]
            relevance: Ground truth relevance [batch_size]
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(relevance, torch.Tensor):
            relevance = relevance.cpu().numpy()
        
        for qid, did, score, rel in zip(query_ids, doc_ids, scores, relevance):
            self.query_results[qid].append((did, float(score), int(rel)))
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute IR evaluation metrics"""
        if not self.query_results:
            return {}
        
        metrics = {}
        
        # Compute metrics for each k
        for k in self.k_values:
            precision_at_k = []
            recall_at_k = []
            ndcg_at_k = []
            
            for query_id, results in self.query_results.items():
                # Sort by score (descending)
                results.sort(key=lambda x: x[1], reverse=True)
                
                # Get top-k results
                top_k = results[:k]
                
                # Precision@k
                relevant_in_k = sum(1 for _, _, rel in top_k if rel > 0)
                precision_at_k.append(relevant_in_k / k if k > 0 else 0)
                
                # Recall@k
                total_relevant = sum(1 for _, _, rel in results if rel > 0)
                recall_at_k.append(relevant_in_k / total_relevant if total_relevant > 0 else 0)
                
                # NDCG@k
                dcg = sum((2**rel - 1) / np.log2(i + 2) for i, (_, _, rel) in enumerate(top_k))
                
                # Ideal DCG
                ideal_rels = sorted([rel for _, _, rel in results], reverse=True)[:k]
                idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
                
                ndcg_at_k.append(dcg / idcg if idcg > 0 else 0)
            
            metrics[f'precision@{k}'] = float(np.mean(precision_at_k))
            metrics[f'recall@{k}'] = float(np.mean(recall_at_k))
            metrics[f'ndcg@{k}'] = float(np.mean(ndcg_at_k))
        
        # Mean Average Precision (MAP)
        map_scores = []
        for query_id, results in self.query_results.items():
            results.sort(key=lambda x: x[1], reverse=True)
            
            relevant_docs = 0
            precision_sum = 0
            
            for i, (_, _, rel) in enumerate(results):
                if rel > 0:
                    relevant_docs += 1
                    precision_sum += relevant_docs / (i + 1)
            
            total_relevant = sum(1 for _, _, rel in results if rel > 0)
            map_scores.append(precision_sum / total_relevant if total_relevant > 0 else 0)
        
        metrics['map'] = float(np.mean(map_scores))
        
        return metrics

class ComplementarityAnalyzer:
    """Analyzer for measuring KG and Dense vector complementarity"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset analysis data"""
        self.kg_predictions = []
        self.dense_predictions = []
        self.fusion_predictions = []
        self.ground_truth = []
        
    def add_batch(self, kg_preds: torch.Tensor, dense_preds: torch.Tensor,
                  fusion_preds: torch.Tensor, labels: torch.Tensor):
        """
        Add batch predictions from different models
        
        Args:
            kg_preds: KG-only model predictions
            dense_preds: Dense-only model predictions  
            fusion_preds: Fusion model predictions
            labels: Ground truth labels
        """
        for preds in [kg_preds, dense_preds, fusion_preds, labels]:
            if isinstance(preds, torch.Tensor):
                preds = preds.cpu().numpy()
        
        self.kg_predictions.extend(kg_preds.flatten())
        self.dense_predictions.extend(dense_preds.flatten())
        self.fusion_predictions.extend(fusion_preds.flatten())
        self.ground_truth.extend(labels.flatten())
    
    def compute_complementarity_metrics(self) -> Dict[str, float]:
        """Compute complementarity analysis metrics"""
        if not self.kg_predictions:
            return {}
        
        kg_preds = np.array(self.kg_predictions)
        dense_preds = np.array(self.dense_predictions)
        fusion_preds = np.array(self.fusion_predictions)
        labels = np.array(self.ground_truth)
        
        # Convert to binary if needed
        kg_binary = (kg_preds > 0.5).astype(int)
        dense_binary = (dense_preds > 0.5).astype(int)
        fusion_binary = (fusion_preds > 0.5).astype(int)
        
        # Individual model accuracies
        kg_acc = accuracy_score(labels, kg_binary)
        dense_acc = accuracy_score(labels, dense_binary)
        fusion_acc = accuracy_score(labels, fusion_binary)
        
        # Complementarity measures
        kg_correct = (kg_binary == labels)
        dense_correct = (dense_binary == labels)
        
        # Cases where KG is correct but Dense is wrong
        kg_only_correct = kg_correct & ~dense_correct
        # Cases where Dense is correct but KG is wrong  
        dense_only_correct = dense_correct & ~kg_correct
        # Cases where both are correct
        both_correct = kg_correct & dense_correct
        # Cases where both are wrong
        both_wrong = ~kg_correct & ~dense_correct
        
        total = len(labels)
        
        metrics = {
            'kg_accuracy': float(kg_acc),
            'dense_accuracy': float(dense_acc),
            'fusion_accuracy': float(fusion_acc),
            'fusion_improvement': float(fusion_acc - max(kg_acc, dense_acc)),
            'kg_only_correct_rate': float(np.sum(kg_only_correct) / total),
            'dense_only_correct_rate': float(np.sum(dense_only_correct) / total),
            'both_correct_rate': float(np.sum(both_correct) / total),
            'both_wrong_rate': float(np.sum(both_wrong) / total),
            'complementarity_score': float(np.sum(kg_only_correct | dense_only_correct) / total)
        }
        
        # Oracle upper bound (if either model is correct)
        oracle_correct = kg_correct | dense_correct
        metrics['oracle_accuracy'] = float(np.sum(oracle_correct) / total)
        metrics['fusion_vs_oracle_gap'] = float(metrics['oracle_accuracy'] - fusion_acc)
        
        return metrics

class EvaluationSuite:
    """Complete evaluation suite for KG + Dense research"""
    
    def __init__(self, output_dir: str = "results/evaluations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.qa_evaluator = QAEvaluator()
        self.ir_evaluator = IREvaluator()
        self.complementarity_analyzer = ComplementarityAnalyzer()
        
        self.results = {}
    
    def evaluate_qa_model(self, model, dataloader, device: str = 'cpu') -> Dict[str, float]:
        """Evaluate QA model on dataset"""
        model.eval()
        self.qa_evaluator.reset()
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Get predictions
                outputs = model(batch, task='qa')
                predictions = torch.sigmoid(outputs).squeeze()
                
                # Add to evaluator
                self.qa_evaluator.add_batch(
                    predictions=predictions,
                    labels=batch['labels'],
                    scores=predictions
                )
        
        return self.qa_evaluator.compute_metrics()
    
    def evaluate_ir_model(self, model, dataloader, device: str = 'cpu') -> Dict[str, float]:
        """Evaluate IR model on dataset"""
        model.eval()
        self.ir_evaluator.reset()
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Get predictions
                outputs = model(batch, task='ir')
                scores = torch.sigmoid(outputs).squeeze()
                
                # Add to evaluator
                self.ir_evaluator.add_batch(
                    query_ids=batch['query_ids'],
                    doc_ids=batch['doc_ids'],
                    scores=scores,
                    relevance=batch['relevance']
                )
        
        return self.ir_evaluator.compute_metrics()
    
    def evaluate_complementarity(self, kg_model, dense_model, fusion_model, 
                                dataloader, task: str = 'qa', device: str = 'cpu') -> Dict[str, float]:
        """Evaluate complementarity between KG and Dense models"""
        kg_model.eval()
        dense_model.eval()
        fusion_model.eval()
        
        self.complementarity_analyzer.reset()
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Get predictions from all models
                kg_outputs = kg_model(batch, task=task)
                dense_outputs = dense_model(batch, task=task)
                fusion_outputs = fusion_model(batch, task=task)
                
                kg_preds = torch.sigmoid(kg_outputs).squeeze()
                dense_preds = torch.sigmoid(dense_outputs).squeeze()
                fusion_preds = torch.sigmoid(fusion_outputs).squeeze()
                
                # Add to analyzer
                labels = batch['labels'] if task == 'qa' else batch['relevance']
                self.complementarity_analyzer.add_batch(
                    kg_preds=kg_preds,
                    dense_preds=dense_preds,
                    fusion_preds=fusion_preds,
                    labels=labels
                )
        
        return self.complementarity_analyzer.compute_complementarity_metrics()
    
    def run_full_evaluation(self, models: Dict[str, Any], dataloaders: Dict[str, Any],
                           device: str = 'cpu') -> Dict[str, Any]:
        """Run complete evaluation suite"""
        results = {
            'qa_results': {},
            'ir_results': {},
            'complementarity_results': {}
        }
        
        # QA Evaluation
        if 'qa_dataloader' in dataloaders:
            logger.info("Evaluating QA models...")
            for model_name, model in models.items():
                if 'qa' in model_name.lower() or 'fusion' in model_name.lower():
                    qa_metrics = self.evaluate_qa_model(model, dataloaders['qa_dataloader'], device)
                    results['qa_results'][model_name] = qa_metrics
                    logger.info(f"{model_name} QA F1: {qa_metrics.get('f1', 0):.4f}")
        
        # IR Evaluation  
        if 'ir_dataloader' in dataloaders:
            logger.info("Evaluating IR models...")
            for model_name, model in models.items():
                if 'ir' in model_name.lower() or 'fusion' in model_name.lower():
                    ir_metrics = self.evaluate_ir_model(model, dataloaders['ir_dataloader'], device)
                    results['ir_results'][model_name] = ir_metrics
                    logger.info(f"{model_name} IR MAP: {ir_metrics.get('map', 0):.4f}")
        
        # Complementarity Analysis
        if all(key in models for key in ['kg_model', 'dense_model', 'fusion_model']):
            logger.info("Analyzing complementarity...")
            
            for task, dataloader_key in [('qa', 'qa_dataloader'), ('ir', 'ir_dataloader')]:
                if dataloader_key in dataloaders:
                    comp_metrics = self.evaluate_complementarity(
                        models['kg_model'], models['dense_model'], models['fusion_model'],
                        dataloaders[dataloader_key], task, device
                    )
                    results['complementarity_results'][task] = comp_metrics
                    logger.info(f"{task.upper()} Complementarity Score: {comp_metrics.get('complementarity_score', 0):.4f}")
        
        # Save results
        self.save_results(results)
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save evaluation results to file"""
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        
        # Also save human-readable summary
        self.save_summary(results, filepath.with_suffix('.txt'))
    
    def save_summary(self, results: Dict[str, Any], filepath: Path):
        """Save human-readable summary"""
        lines = [
            "=" * 60,
            "KG + DENSE VECTOR COMPLEMENTARITY EVALUATION RESULTS",
            "=" * 60,
            ""
        ]
        
        # QA Results
        if results.get('qa_results'):
            lines.extend([
                "QUESTION ANSWERING RESULTS:",
                "-" * 30
            ])
            for model_name, metrics in results['qa_results'].items():
                lines.append(f"{model_name}:")
                for metric, value in metrics.items():
                    lines.append(f"  {metric}: {value:.4f}")
                lines.append("")
        
        # IR Results
        if results.get('ir_results'):
            lines.extend([
                "INFORMATION RETRIEVAL RESULTS:",
                "-" * 35
            ])
            for model_name, metrics in results['ir_results'].items():
                lines.append(f"{model_name}:")
                for metric, value in metrics.items():
                    lines.append(f"  {metric}: {value:.4f}")
                lines.append("")
        
        # Complementarity Results
        if results.get('complementarity_results'):
            lines.extend([
                "COMPLEMENTARITY ANALYSIS:",
                "-" * 25
            ])
            for task, metrics in results['complementarity_results'].items():
                lines.append(f"{task.upper()} Task:")
                for metric, value in metrics.items():
                    lines.append(f"  {metric}: {value:.4f}")
                lines.append("")
        
        lines.append("=" * 60)
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

if __name__ == "__main__":
    # Test evaluation framework
    print("Testing evaluation framework...")
    
    evaluator = EvaluationSuite()
    
    # Test QA evaluator
    qa_eval = QAEvaluator()
    qa_eval.add_batch(
        predictions=torch.tensor([0.8, 0.3, 0.9, 0.1]),
        labels=torch.tensor([1, 0, 1, 0]),
        scores=torch.tensor([0.8, 0.3, 0.9, 0.1])
    )
    qa_metrics = qa_eval.compute_metrics()
    print(f"QA Metrics: {qa_metrics}")
    
    # Test IR evaluator
    ir_eval = IREvaluator()
    ir_eval.add_batch(
        query_ids=['q1', 'q1', 'q2', 'q2'],
        doc_ids=['d1', 'd2', 'd3', 'd4'],
        scores=torch.tensor([0.9, 0.7, 0.8, 0.6]),
        relevance=torch.tensor([1, 0, 1, 1])
    )
    ir_metrics = ir_eval.compute_metrics()
    print(f"IR Metrics: {ir_metrics}")
    
    print("✓ Evaluation framework tested successfully!")
