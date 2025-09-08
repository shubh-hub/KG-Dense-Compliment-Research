#!/usr/bin/env python3
"""
Baseline Complementarity Training Experiments
Systematic evaluation of Dense-only, KG-only, and all 4 fusion architectures
"""

import argparse
import json
import numpy as np
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from math import comb 
# McNemar test implementation (scipy version compatibility)
try:
    from scipy.stats import mcnemar
except ImportError:
    try:
        from scipy.stats.contingency import mcnemar
    except ImportError:
        # Fallback implementation
        def mcnemar(table):
            """Simple McNemar test implementation"""
            b, c = table[0, 1], table[1, 0]
            if b + c == 0:
                return type('obj', (object,), {'pvalue': 1.0})()
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            from scipy.stats import chi2 as chi2_dist
            pvalue = 1 - chi2_dist.cdf(chi2, 1)
            return type('obj', (object,), {'pvalue': pvalue})()

from models.baseline_architectures import (
    DenseVectorEncoder, 
    KnowledgeGraphEncoder,
    KGDenseFusionModel,
    get_model_config as get_baseline_config
)

def _collect_val_logits_labels_slices(model, dataloader, global_edge_index, global_edge_type, device=None):
    """Run model on a dataloader and collect raw logits, labels, and slice labels."""
    if device is None:
        device = get_device()
    model = model.to(device)
    global_edge_index = global_edge_index.to(device)
    global_edge_type = global_edge_type.to(device)

    model.eval()
    logits_all = []
    labels_all = []
    slices_all = []

    with torch.no_grad():
        for input_ids, attention_mask, entity_ids, labels, slice_labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            entity_ids = entity_ids.to(device)
            labels = labels.to(device).float()

            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'entity_ids': entity_ids,
                'edge_index': global_edge_index,
                'edge_type': global_edge_type
            }
            out = model(batch, task='qa')
            if isinstance(out, tuple):
                out = out[0]
            logits_all.append(out.detach().view(-1).cpu())
            labels_all.append(labels.detach().view(-1).cpu())
            slices_all.append(slice_labels.detach().view(-1).cpu())

    logits = torch.cat(logits_all, dim=0)        # [N]
    labels = torch.cat(labels_all, dim=0).float() # [N]
    slices = torch.cat(slices_all, dim=0).long() # [N]
    return logits, labels, slices


def _fit_platt_scaler(logits: torch.Tensor, labels: torch.Tensor, steps: int = 200, lr: float = 0.1):
    """
    Fit a Platt scaler (sigmoid(a * logit + b)) on validation logits via Adam.
    Runs on CPU for stability; super fast for N~200.
    """
    X = logits.detach().cpu()
    y = labels.detach().cpu()
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(a * X + b, y)
        loss.backward()
        opt.step()
    return float(a.detach().item()), float(b.detach().item())

# Set M1 optimizations
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model_config(config_name='m1_optimized'):
    """Get model configuration for experiments - uses baseline_architectures configs"""
    return get_baseline_config(config_name)

def get_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def create_synthetic_dataset_v2(num_samples=1000, seq_len=32, num_entities=500, num_relations=20, batch_size=32, split_data=True, seed=42):
    """
    Create synthetic dataset v2 with learnable complementarity signals
    
    4 slices with visible cues:
    - Dense slice: special bigram tokens + synonym patterns
    - KG slice: 2-3 hop typed motifs around entities  
    - Both slice: requires both signals for correct answer
    - None slice: neither signal helps (random baseline)
    """
    torch.manual_seed(seed)
    device = get_device()
    
    # Vocabulary setup (DistilBERT/bert vocab size = 30522)
    # Put synthetic bigrams high in the vocab to avoid [CLS]=101 and [SEP]=102 collisions
    VOCAB_SIZE = 30522
    BIGRAM_BASE = 29000
    SPECIAL_BIGRAMS = [
        (BIGRAM_BASE + 0, BIGRAM_BASE + 1),
        (BIGRAM_BASE + 2, BIGRAM_BASE + 3),
        (BIGRAM_BASE + 4, BIGRAM_BASE + 5),
        (BIGRAM_BASE + 6, BIGRAM_BASE + 7),
    ]  # 2 positive, 2 negative bigrams
    SYNONYM_PAIRS = [
        (BIGRAM_BASE + 8, BIGRAM_BASE + 9),
        (BIGRAM_BASE + 10, BIGRAM_BASE + 11),
        (BIGRAM_BASE + 12, BIGRAM_BASE + 13),
        (BIGRAM_BASE + 14, BIGRAM_BASE + 15)
    ]  # Dense synonyms
    
    # Create global graph structure (single edge_index for all samples)
    global_edges = []
    global_edge_types = []
    
    # Precompute entities that have outgoing edges with (edge_type % 4) < 2 for BOTH slice
    # This ensures we can force KG-positive entities without mutating global edges
    
    # Create typed motifs for KG cues
    KG_MOTIFS = [
        # 2-hop motifs: entity -> relation -> entity -> relation -> target
        [(0, 1, 2), (1, 2, 3)],  # person -> born_in -> place -> located_in -> country
        [(0, 4, 5), (5, 6, 7)],  # work -> created_by -> person -> nationality -> country  
        [(0, 8, 9), (9, 10, 11)], # event -> happened_in -> year -> part_of -> decade
        [(0, 12, 13), (13, 14, 15)] # object -> made_of -> material -> found_in -> location
    ]
    
    # Build global graph with typed relations
    for i in range(num_entities - 1):
        for j in range(min(3, num_entities - i - 1)):  # Each entity connects to 3 others
            rel_type = (i + j) % num_relations
            global_edges.append([i, i + j + 1])
            global_edge_types.append(rel_type)
            # Add reverse edges for some relations
            if rel_type < num_relations // 2:
                global_edges.append([i + j + 1, i])
                global_edge_types.append(rel_type + num_relations // 2)
    
    global_edge_index = torch.tensor(global_edges, dtype=torch.long).t().contiguous()
    global_edge_type = torch.tensor(global_edge_types, dtype=torch.long)
    
    # balanced, deterministic subset of "good" entities (~50%)
    rng = random.Random(seed)
    good_entities = rng.sample(range(num_entities), k=num_entities // 2)
    bad_entities = sorted(set(range(num_entities)) - set(good_entities))
    
    print(f"  Graph edges: {global_edge_index.size(1)}   Relation types: {num_relations}")
    print(f"  Good entities for BOTH slice: {len(good_entities)}")
    
    # Generate samples for each slice
    samples_per_slice = num_samples // 4
    
    input_ids_list = []
    attention_mask_list = []
    entity_ids_list = []
    labels_list = []
    slice_labels = []  # Track which slice each sample belongs to
    
    for slice_idx in range(4):
        for _ in range(samples_per_slice):
            # Create input sequence with synthetic bigrams
            input_ids = torch.zeros(4, dtype=torch.long)
            attention_mask = torch.zeros(4, dtype=torch.long)
            
            # Add CLS and SEP tokens for proper BERT-style input
            CLS_ID, SEP_ID = 101, 102  # BERT's [CLS], [SEP]
            input_ids[0] = CLS_ID
            attention_mask[0] = 1
            
            # Add bigram tokens based on slice
            if slice_idx == 0:  # Dense slice  ➜ make KG misleading, label from dense cue only
                # Make dense cue more learnable with 80% positive bias
                if torch.rand(1).item() < 0.8:
                    bigram_idx = random.choice([0, 2])  # Positive bigrams
                else:
                    bigram_idx = random.choice([1, 3])  # Negative bigrams
                bigram = SPECIAL_BIGRAMS[bigram_idx]
                input_ids[1:3] = torch.tensor(bigram)  # After CLS token
                attention_mask[1:3] = 1
                
                # Dense cue determines label
                label = bigram_idx % 2 == 0  # Even indices are positive
                # Pick entity from "bad" bucket most of the time so KG-only tends to be wrong here
                if torch.rand(1).item() < 0.8 and bad_entities:
                    entity_id = random.choice(bad_entities)
                else:
                    entity_id = random.randint(0, num_entities - 1)
                slice_labels.append(0)
                # Done for dense slice
                input_ids_list.append(input_ids); attention_mask_list.append(attention_mask)
                entity_ids_list.append(entity_id); labels_list.append(label)
                continue
            else:
                # Fill with random tokens for non-dense slices (nowhere near special bigrams or [CLS]/[SEP])
                random_tokens = torch.randint(1000, 5000, (2,))
                input_ids[1:3] = random_tokens  # After CLS token
                attention_mask[1:3] = 1
            
            # Add SEP token at the end
            input_ids[3] = SEP_ID
            attention_mask[3] = 1
            
            # Entity selection and labeling based on slice type
            if slice_idx == 1:  # KG slice  ➜ make Dense misleading, label from KG only
                # Force a NEGATIVE dense cue (choose a negative bigram)
                neg_idx = random.choice([1, 3])
                neg_bigram = SPECIAL_BIGRAMS[neg_idx]
                input_ids[1:3] = torch.tensor(neg_bigram)
                attention_mask[1:3] = 1
                # Choose entity to set label; clearly positive-leaning to make KG learnable
                if torch.rand(1).item() < 0.80 and good_entities:
                    entity_id = random.choice(good_entities); label = 1.0
                else:
                    entity_id = random.choice(bad_entities if bad_entities else list(range(num_entities))); label = 0.0
                slice_labels.append(1)
                
            elif slice_idx == 2:  # BOTH slice - requires both dense AND KG signals (ADVERSARIAL)
                # Force dense cue to be POSITIVE (100% ON) - overwrite the random tokens
                even_choices = [0, 2]  # Positive bigram indices
                bigram_idx = random.choice(even_choices)
                bigram = SPECIAL_BIGRAMS[bigram_idx]
                input_ids[1:3] = torch.tensor(bigram)  # Overwrite with positive bigram
                
                # Force KG cue to be POSITIVE by picking entity from good_entities
                if len(good_entities) > 0:
                    entity_id = random.choice(good_entities)
                else:
                    entity_id = random.randint(0, num_entities - 1)
                
                # Anti-AND: both cues ON ⇒ label = 0; this forces complementarity
                label = 0.0
                slice_labels.append(2)
                
            else:  # None slice (neutral baseline)
                entity_id = random.randint(0, num_entities - 1)
                # Neutral: 50/50 so models can't farm easy negatives
                label = float(torch.rand(1).item() > 0.5)
                slice_labels.append(3)
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            entity_ids_list.append(entity_id)
            labels_list.append(label)
    
    # Convert to tensors
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    entity_ids = torch.tensor(entity_ids_list)
    labels = torch.tensor(labels_list, dtype=torch.float32)
    slice_labels = torch.tensor(slice_labels, dtype=torch.long)
    
    # Quick sanity: slice positive rates
    with torch.no_grad():
        pr = []
        for s in range(4):
            m = (slice_labels == s)
            pr.append(float(labels[m].mean().item()) if m.any() else float('nan'))
        print(f"  Slice positive rates (dense/kg/both/none): {pr[0]:.2f} / {pr[1]:.2f} / {pr[2]:.2f} / {pr[3]:.2f}")

    if split_data:
        # Create stratified 80/10/10 train/val/test split by slice
        dataset = TensorDataset(input_ids, attention_mask, entity_ids, labels, slice_labels)
        
        # Stratify by slice to preserve proportions
        indices = torch.arange(len(slice_labels))
        by_slice = {s: indices[slice_labels == s] for s in range(4)}
        
        train_idx, val_idx, test_idx = [], [], []
        for s, idx in by_slice.items():
            n = len(idx)
            perm = idx[torch.randperm(n)]
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            train_idx.append(perm[:n_train])
            val_idx.append(perm[n_train:n_train+n_val])
            test_idx.append(perm[n_train+n_val:])
        
        train_indices = torch.cat(train_idx)
        val_indices = torch.cat(val_idx)
        test_indices = torch.cat(test_idx)
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        meta = {
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
            'slice_labels_train': slice_labels[train_indices],
            'slice_labels_val': slice_labels[val_indices],
            'slice_labels_test': slice_labels[test_indices]
        }
        
        return train_dataloader, val_dataloader, test_dataloader, global_edge_index, global_edge_type, meta
    else:
        # Return full dataset as single dataloader
        dataset = TensorDataset(input_ids, attention_mask, entity_ids, labels, slice_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        meta = {
            'total_size': len(dataset),
        }
        
        return dataloader, global_edge_index, global_edge_type, meta

def init_qa_bias(model, train_loader):
    """Initialize classifier bias to the base rate"""
    import math
    total = pos = 0
    for _, _, _, y, _ in train_loader:
        pos += y.sum().item()
        total += y.numel()
    prior = min(max(pos/total, 1e-3), 1-1e-3)
    b = math.log(prior/(1-prior))
    with torch.no_grad():
        head = model.qa_head[-1] if isinstance(model.qa_head, nn.Sequential) else model.qa_head
        head.bias.fill_(b)

def train_model(model, train_loader, val_loader, global_edge_index, global_edge_type, num_epochs=3, lr=1e-4, model_name="Model"):
    """Train model and return training metrics"""
    device = get_device()
    model = model.to(device)
    global_edge_index = global_edge_index.to(device)
    global_edge_type = global_edge_type.to(device)
    
    # Different learning rates for KG and other parameters
    kg_params, other_params = [], []
    for n, p in model.named_parameters():
        (kg_params if ("kg_encoder" in n) else other_params).append(p)
    
    optimizer = torch.optim.AdamW(
        [{'params': other_params, 'lr': 1e-4},
         {'params': kg_params,   'lr': 3e-4}],  # Higher LR for KG
        weight_decay=0.01
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # Check if this is a hierarchical gating fusion model
    is_hierarchical_gating = hasattr(model, 'fusion_type') and model.fusion_type == 'hierarchical_gating'
    
    model.train()
    epoch_losses = []
    epoch_accuracies = []
    
    print(f"  Training {model_name} model...")
    
    # Initialize classifier bias to base rate
    init_qa_bias(model, train_loader)
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (input_ids, attention_mask, entity_ids, labels, slice_labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
            # Move to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            entity_ids = entity_ids.to(device)
            labels = labels.to(device).float()
            slice_labels = slice_labels.to(device)
            
            # Create batch dictionary with global graph structure
            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'entity_ids': entity_ids,
                'edge_index': global_edge_index,
                'edge_type': global_edge_type
            }
            
            # Forward pass
            optimizer.zero_grad()
            
            if is_hierarchical_gating:
                # Get auxiliary outputs for supervision
                result = model(batch, task='qa', return_aux_outputs=True)
                if isinstance(result, tuple):
                    outputs, aux_outputs = result
                else:
                    outputs = result
                    aux_outputs = None
                
                # Class-balanced loss weighting to prevent "always 0" degenerate solution
                slice_weights = torch.ones_like(labels)
                # Up-weight positive examples in Dense/KG slices to balance classes
                slice_weights[(slice_labels == 0) & (labels == 1)] = 3.0  # Dense positives (increased)
                slice_weights[(slice_labels == 1) & (labels == 1)] = 1.5  # KG positives
                slice_weights[slice_labels == 2] = 2.5  # Up-weight BOTH slice (all negatives)
                
                # Main task loss with class balancing
                loss_vec = F.binary_cross_entropy_with_logits(outputs.squeeze(), labels, reduction='none')
                loss = (loss_vec * slice_weights).mean()
                
                # Enforce negative margins on BOTH (2) and NONE (3) for fusion models
                apply_margins = 'hierarchical_gating' in model.__class__.__name__.lower() or 'fusion' in model_name.lower()
                if apply_margins:
                    logits_s = outputs.squeeze()
                    margin = 2.0  # Stronger margin
                    coeff = 0.6   # Higher coefficient for margin loss
                    for bad_slice in (2, 3):  # BOTH and NONE slices
                        m = (slice_labels == bad_slice)
                        if m.any():
                            loss = loss + coeff * F.relu(logits_s[m] + margin).mean()
                
                # Gate supervision loss for 3-way gating (kg, dense, null)
                if aux_outputs is not None:
                    kg_gate = aux_outputs['kg_gate'].squeeze()
                    dense_gate = aux_outputs['dense_gate'].squeeze()
                    null_gate = aux_outputs.get('null_gate', torch.zeros_like(kg_gate))
                    
                    # Gate targets based on slice (kg, dense, null)
                    kg_target = torch.zeros_like(kg_gate)
                    dense_target = torch.zeros_like(dense_gate)
                    null_target = torch.zeros_like(null_gate)
                    
                    for i, slice_label in enumerate(slice_labels):
                        if slice_label == 0:  # Dense slice
                            kg_target[i], dense_target[i], null_target[i] = 0.05, 0.90, 0.05
                        elif slice_label == 1:  # KG slice
                            kg_target[i], dense_target[i], null_target[i] = 0.90, 0.05, 0.05
                        else:  # BOTH or NONE slice -> prefer NULL
                            kg_target[i], dense_target[i], null_target[i] = 0.10, 0.10, 0.80
                    
                    gate_loss = (F.mse_loss(kg_gate, kg_target) + 
                               F.mse_loss(dense_gate, dense_target) +
                               F.mse_loss(null_gate, null_target))
                    loss = loss + 0.6 * gate_loss  # Strong gate supervision
                    
                    # Oracle distillation loss
                    if 'kg_aux_logits' in aux_outputs and 'dense_aux_logits' in aux_outputs:
                        kg_aux = aux_outputs['kg_aux_logits'].squeeze()
                        dense_aux = aux_outputs['dense_aux_logits'].squeeze()
                        
                        oracle_loss = (F.binary_cross_entropy_with_logits(kg_aux, labels) +
                                     F.binary_cross_entropy_with_logits(dense_aux, labels))
                        loss = loss + 0.1 * oracle_loss
                    
                    # Entropy regularization for decisive routing
                    gate_probs = torch.stack([kg_gate, dense_gate], dim=1)
                    gate_entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=1)
                    
                    # Encourage low entropy (decisive) on single slices, high entropy on both/none
                    entropy_targets = torch.zeros_like(gate_entropy)
                    for i, slice_label in enumerate(slice_labels):
                        if slice_label in [0, 1]:  # Single slices - want decisive routing
                            entropy_targets[i] = 0.1  # Low entropy target
                        else:  # BOTH/NONE slices - want balanced routing
                            entropy_targets[i] = 0.693  # log(2) = max entropy for 2 options
                    
                    entropy_loss = F.mse_loss(gate_entropy, entropy_targets)
                    loss = loss + 0.1 * entropy_loss
            else:
                # Class-balanced loss for all models to encourage specialization
                outputs = model(batch, task='qa')
                
                # Stronger slice-specific weighting to force specialization
                slice_weights = torch.ones_like(labels)
                
                # Dense model: heavily weight dense slice performance
                if 'dense' in model.__class__.__name__.lower():
                    slice_weights[slice_labels == 0] = 3.0  # Dense slice
                    slice_weights[slice_labels == 1] = 0.5  # KG slice (de-emphasize)
                # KG model: heavily weight KG slice performance  
                elif 'kg' in model.__class__.__name__.lower():
                    slice_weights[slice_labels == 0] = 0.5  # Dense slice (de-emphasize)
                    slice_weights[slice_labels == 1] = 3.5  # KG slice (increased from 3.0)
                else:
                    # Fusion models: balanced weighting
                    slice_weights[(slice_labels == 0) & (labels == 1)] = 2.0  # Dense positives
                    slice_weights[(slice_labels == 1) & (labels == 1)] = 2.0  # KG positives
                    slice_weights[slice_labels == 2] = 2.5  # BOTH slice
                
                loss_vec = F.binary_cross_entropy_with_logits(outputs.squeeze(), labels, reduction='none')
                loss = (loss_vec * slice_weights).mean()
                
                # Enforce negative margins on BOTH (2) and NONE (3) for fusion models
                apply_margins = 'hierarchical_gating' in model.__class__.__name__.lower() or 'fusion' in model_name.lower()
                if apply_margins:
                    logits_s = outputs.squeeze()
                    margin = 2.0  # Stronger margin
                    coeff = 0.6   # Higher coefficient for margin loss
                    for bad_slice in (2, 3):  # BOTH and NONE slices
                        m = (slice_labels == bad_slice)
                        if m.any():
                            loss = loss + coeff * F.relu(logits_s[m] + margin).mean()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.sigmoid(outputs.squeeze()) > 0.5
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct_predictions / total_predictions
        
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)
        
        print(f"    Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
    
    # Minimal, defined training diagnostics
    return {
        'losses': epoch_losses,
        'train_accuracies': epoch_accuracies,
        'final_train_acc': epoch_accuracies[-1] if epoch_accuracies else 0.0
    }

def compute_optimal_threshold(model, dataloader, global_edge_index, global_edge_type, device=None, calibrate=True):
    """
    Compute optimal threshold on validation set.
    - Optional Platt scaling (calibrate=True) to fix miscalibration.
    - Slice-aware objective: F1 for dense/kg/none, TNR for both.
    - Uses a safer mid-range threshold grid to avoid pathological 0.05/0.95 picks.
    Returns: (best_threshold, platt_dict_or_None)
    """
    if device is None:
        device = get_device()

    # 1) Collect raw logits/labels/slices on val
    logits_t, labels_t, slices_t = _collect_val_logits_labels_slices(
        model, dataloader, global_edge_index, global_edge_type, device=device
    )

    # 2) Optional Platt scaling on validation
    if calibrate:
        a, b = _fit_platt_scaler(logits_t, labels_t)
        probs = torch.sigmoid(a * logits_t + b).numpy()
        platt = {'a': a, 'b': b}
    else:
        probs = torch.sigmoid(logits_t).numpy()
        platt = None

    labels = labels_t.numpy().astype(int)
    slices = slices_t.numpy().astype(int)

    # 3) Threshold search (mid-range grid)
    best_threshold, best_score = 0.5, -1.0
    grid = np.linspace(0.10, 0.90, 17)  # avoid 0.05/0.95 extremes

    for threshold in grid:
        preds = (probs > threshold).astype(int)
        per_slice = []
        for s in range(4):  # dense=0, kg=1, both=2, none=3
            m = (slices == s)
            if not np.any(m):
                continue
            y_s, p_s = labels[m], preds[m]
            if s == 2:  # BOTH: use TNR
                tn = np.sum((p_s == 0) & (y_s == 0))
                fp = np.sum((p_s == 1) & (y_s == 0))
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                per_slice.append(tnr)
            else:       # others: F1
                tp = np.sum((p_s == 1) & (y_s == 1))
                fp = np.sum((p_s == 1) & (y_s == 0))
                fn = np.sum((p_s == 0) & (y_s == 1))
                denom = (2 * tp + fp + fn)
                f1 = (2 * tp) / denom if denom > 0 else 0.0
                per_slice.append(f1)
        score = float(np.mean(per_slice)) if per_slice else 0.0
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, platt


def evaluate_model(model, dataloader, global_edge_index, global_edge_type, threshold=0.5, platt=None, return_probs=False):
    """
    Evaluate model and return predictions, labels, and slice labels.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        global_edge_index: Global edge indices for KG
        global_edge_type: Global edge types for KG
        threshold: Decision threshold (default: 0.5)
        platt: Optional dict with 'a' and 'b' for Platt scaling
        return_probs: If True, return probabilities instead of binary predictions
        
    Returns:
        Tuple of (predictions, labels, slice_labels) or (probs, labels, slice_labels) if return_probs
    """
    device = get_device()
    model = model.to(device)
    global_edge_index = global_edge_index.to(device)
    global_edge_type = global_edge_type.to(device)
    
    model.eval()
    all_probs, all_labels, all_slice_labels = [], [], []
    
    with torch.no_grad():
        for input_ids, attention_mask, entity_ids, labels, slice_labels in dataloader:
            # Move batch to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            entity_ids = entity_ids.to(device)
            
            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'entity_ids': entity_ids,
                'edge_index': global_edge_index,
                'edge_type': global_edge_type
            }
            
            # Get model outputs
            out = model(batch, task='qa')
            if isinstance(out, tuple):
                out = out[0]  # Handle case where model returns (logits, ...)
                
            # Apply Platt scaling if provided
            if platt is not None:
                a, b = float(platt['a']), float(platt['b'])
                probs = torch.sigmoid(a * out.view(-1) + b)
            else:
                probs = torch.sigmoid(out.view(-1))
            
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_slice_labels.append(slice_labels.cpu())
    
    # Convert to numpy arrays
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_slice_labels = torch.cat(all_slice_labels).numpy()
    
    if return_probs:
        return all_probs, all_labels, all_slice_labels
    
    # Convert probabilities to binary predictions
    predictions = (all_probs > threshold).astype(int)
    return predictions, all_labels, all_slice_labels

def summarize_gates(model, dataloader, edge_index, edge_type):
    """Summarize gate behavior by slice to verify routing with 3-way gating"""
    device = next(model.parameters()).device
    model.eval()
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    slice_names = {0: 'dense', 1: 'kg', 2: 'both', 3: 'none'}
    sums = {k: {'kg': 0.0, 'dense': 0.0, 'null': 0.0, 'n': 0} for k in slice_names}
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, entity_ids, labels, slice_labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            entity_ids = entity_ids.to(device)
            
            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'entity_ids': entity_ids,
                'edge_index': edge_index,
                'edge_type': edge_type
            }
            
            # Get gate values
            result = model(batch, task='qa', return_aux_outputs=True)
            if isinstance(result, tuple) and len(result) > 1:
                _, aux = result
                kg_gate = aux['kg_gate'].squeeze().cpu().numpy()
                dense_gate = aux['dense_gate'].squeeze().cpu().numpy()
                null_gate = aux.get('null_gate', torch.zeros_like(aux['kg_gate'])).squeeze().cpu().numpy()
                
                # Update statistics by slice
                for i, sl in enumerate(slice_labels):
                    k = sl.item()
                    sums[k]['kg'] += kg_gate[i]
                    sums[k]['dense'] += dense_gate[i]
                    sums[k]['null'] += null_gate[i]
                    sums[k]['n'] += 1
    
    # Print summary
    print("\nGate Behavior by Slice:")
    for k, v in sorted(sums.items()):
        if v['n'] > 0:
            line = f"  {slice_names[k]:5s} gates → kg={v['kg']/v['n']:.3f}, dense={v['dense']/v['n']:.3f}"
            if 'null' in v:
                line += f", null={v['null']/v['n']:.3f}"
            line += f"  (n={v['n']})"
            print(line)
    
    return sums

def run_complementarity_experiments(num_samples=1000, num_epochs=3, seed=42, calibrate_thresholds=True):
    """
    Run comprehensive complementarity experiments with all model types
    
    Args:
        num_samples: Number of samples to generate
        num_epochs: Number of training epochs
        seed: Random seed for reproducibility
        calibrate_thresholds: Whether to use Platt scaling for calibration
    """
    set_seeds(seed)
    torch.set_float32_matmul_precision("high")
    
    # Track timing for performance analysis
    start_time = time.time()
    
    print("🚀 Starting Baseline Complementarity Training Experiments")
    print("=" * 60)
    
    # M1-optimized configuration
    config = get_model_config('m1_optimized')
    
    # Create synthetic dataset with explicit train/val/test split
    print("📊 Creating synthetic dataset v2 with learnable signals...")
    train_dataloader, val_dataloader, test_dataloader, global_edge_index, global_edge_type, meta = create_synthetic_dataset_v2(
        num_samples=num_samples, split_data=True
    )
    print(f"  Train samples: {meta['train_size']}   Val: {meta['val_size']}   Test: {meta['test_size']}")
    print(f"  Graph edges: {global_edge_index.size(1)}   Relation types: {global_edge_type.max().item() + 1}")
    
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
            # Create model directly
            if model_key == 'dense_only':
                # Create wrapper for Dense-Only model
                class DenseOnlyModel(nn.Module):
                    def __init__(self, config):
                        super().__init__()
                        self.encoder = DenseVectorEncoder(
                            model_name=config.get('dense_model', 'distilbert-base-uncased'),
                            hidden_dim=config.get('hidden_dim', 384),
                            freeze_lm=config.get('freeze_lm', False),
                            pooling=config.get('pooling', 'cls')
                        )
                        self.qa_head = nn.Linear(config.get('hidden_dim', 384), 1)
                    
                    def forward(self, batch, task='qa', return_aux_outputs=False):
                        embeddings = self.encoder(batch['input_ids'], batch['attention_mask'])
                        logits = self.qa_head(embeddings)
                        return logits
                
                model = DenseOnlyModel(config)
                
            elif model_key == 'kg_only':
                # Create wrapper for KG-Only model
                class KGOnlyModel(nn.Module):
                    def __init__(self, config):
                        super().__init__()
                        self.encoder = KnowledgeGraphEncoder(
                            num_entities=config.get('num_entities', 500),
                            num_relations=config.get('num_relations', 20),
                            hidden_dim=config.get('hidden_dim', 384),
                            gnn_type=config.get('gnn_type', 'rgcn'),
                            num_layers=config.get('gnn_layers', 2),
                            num_bases=config.get('num_bases', 16),
                            dropout=config.get('dropout', 0.1)
                        )
                        self.qa_head = nn.Linear(config.get('hidden_dim', 384), 1)
                    
                    def forward(self, batch, task='qa', return_aux_outputs=False):
                        embeddings = self.encoder(batch['entity_ids'], batch['edge_index'], batch.get('edge_type'))
                        logits = self.qa_head(embeddings)
                        return logits
                
                model = KGOnlyModel(config)
                
            else:  # fusion models
                fusion_type = model_key.replace('fusion_', '')
                fusion_config = config.copy()
                fusion_config['fusion_type'] = fusion_type
                model = KGDenseFusionModel(fusion_config)
            
            # Train model
            train_results = train_model(
                model, train_dataloader, val_dataloader, global_edge_index, global_edge_type,
                num_epochs=num_epochs, model_name=model_name
            )
            
            # Find optimal threshold on validation set with optional Platt scaling
            print(f"\n  Computing optimal threshold for {model_name}...")
            best_threshold, platt_params = compute_optimal_threshold(
                model=model,
                dataloader=val_dataloader,
                global_edge_index=global_edge_index,
                global_edge_type=global_edge_type,
                calibrate=calibrate_thresholds
            )
            
            # Evaluate on test set with calibrated threshold
            print(f"  Evaluating {model_name}...")
            predictions, labels, slice_labels = evaluate_model(
                model=model,
                dataloader=test_dataloader,
                global_edge_index=global_edge_index,
                global_edge_type=global_edge_type,
                threshold=best_threshold,
                platt=platt_params
            )
            
            # Get probabilities for more detailed analysis
            probs, _, _ = evaluate_model(
                model=model,
                dataloader=test_dataloader,
                global_edge_index=global_edge_index,
                global_edge_type=global_edge_type,
                platt=platt_params,
                return_probs=True
            )
            
            # Calculate metrics
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions)
            
            # Store results with calibration info
            results[model_key] = {
                'accuracy': accuracy,
                'f1': f1,
                'threshold': best_threshold,
                'calibration': {
                    'platt_a': platt_params.get('a', 1.0) if platt_params else 1.0,
                    'platt_b': platt_params.get('b', 0.0) if platt_params else 0.0,
                    'calibrated': calibrate_thresholds and platt_params is not None
                },
                'predictions': predictions.tolist(),
                'probabilities': probs.tolist(),
                'labels': labels.tolist(),
                'slice_labels': slice_labels.tolist(),
                'train_metrics': train_results
            }
            
            # Add gate diagnostics for hierarchical model
            if model_key == 'fusion_hierarchical_gating':
                print(f"  Gate diagnostics for {model_name}:")
                summarize_gates(model, val_dataloader, global_edge_index, global_edge_type)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"    Accuracy: {accuracy:.4f}")
            print(f"    F1 Score: {f1:.4f}")
            print(f"    Parameters: {total_params:,} total, {trainable_params:,} trainable")
            
            # Compute per-slice metrics
            by_slice = {}
            slice_names = ['dense', 'kg', 'both', 'none']
            print(f"    Per-slice performance:")
            for slice_idx in range(4):
                mask = slice_labels == slice_idx
                if np.sum(mask) > 0:
                    slice_preds = predictions[mask]
                    slice_true = labels[mask]
                    slice_acc = np.mean(slice_preds == slice_true)
                    
                    # Use balanced accuracy for BOTH slice (all negatives by design)
                    if slice_names[slice_idx] == 'both':
                        # Balanced accuracy = (TPR + TNR) / 2
                        tp = np.sum((slice_preds == 1) & (slice_true == 1))
                        tn = np.sum((slice_preds == 0) & (slice_true == 0))
                        fp = np.sum((slice_preds == 1) & (slice_true == 0))
                        fn = np.sum((slice_preds == 0) & (slice_true == 1))
                        
                        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        balanced_acc = (tpr + tnr) / 2.0
                        
                        by_slice[slice_names[slice_idx]] = {
                            "n": int(np.sum(mask)),
                            "acc": float(slice_acc),
                            "balanced_acc": float(balanced_acc),
                        }
                        print(f"      - {slice_names[slice_idx]:5s}: n={np.sum(mask):4d} | acc={slice_acc:.3f} | bal_acc={balanced_acc:.3f}")
                    else:
                        slice_f1 = f1_score(slice_true, slice_preds, zero_division=0)
                        by_slice[slice_names[slice_idx]] = {
                            "n": int(np.sum(mask)),
                            "acc": float(slice_acc),
                            "f1": float(slice_f1),
                        }
                        print(f"      - {slice_names[slice_idx]:5s}: n={np.sum(mask):4d} | acc={slice_acc:.3f} | f1={slice_f1:.3f}")
            
            # Store results in a single consolidated format
            results[model_key] = {
                'status': 'success',
                'model_name': model_name,
                'accuracy': float(accuracy),
                'f1': float(f1),
                'threshold': float(best_threshold),
                'calibration': {
                    'platt_a': platt_params.get('a', 1.0) if platt_params else 1.0,
                    'platt_b': platt_params.get('b', 0.0) if platt_params else 0.0,
                    'calibrated': bool(calibrate_thresholds and platt_params is not None)
                },
                'predictions': predictions.tolist(),
                'probabilities': probs.tolist(),
                'labels': labels.tolist(),
                'slice_labels': slice_labels.tolist(),
                'by_slice': by_slice,
                'training_history': train_results,
                'parameters': int(total_params),
                'trainable_parameters': int(trainable_params),
            }
            
        except Exception as e:
            err = str(e)
            print(f"  ❌ {model_name}: Failed - {err}")
            fail_payload = {
                'status': 'failed',
                'error': err,
                'f1': 0.0,
                'accuracy': 0.0,
                'predictions': [],
                'labels': [],
                'slice_labels': [],
                'by_slice': {},
                'eval_results': {'accuracy': 0.0, 'f1': 0.0, 'predictions': [], 'labels': [], 'by_slice': {}},
                'training_history': {},
                'threshold': 0.5,
                'parameters': 0
            }
            results[model_key] = fail_payload
            results[model_name] = fail_payload
    
    # Enhanced complementarity analysis with McNemar tests
    print(f"\n🔍 Running Enhanced Complementarity Analysis...")
    print("-" * 60)
    
    successful_models = {k: v for k, v in results.items() if v.get('status') == 'success'}
    
    if len(successful_models) >= 3:  # Need at least dense, kg, and one fusion
        # Standard complementarity analysis
        complementarity_analysis = analyze_complementarity(successful_models)
        results['complementarity_analysis'] = complementarity_analysis
        
        # McNemar test between best fusion and baselines
        mcnemar_results = {}
        fusion_models = [k for k in successful_models if k.startswith('fusion_')]
        
        if fusion_models and 'dense_only' in successful_models and 'kg_only' in successful_models:
            # Just compare best fusion vs baselines
            best_fusion = max(fusion_models, key=lambda k: successful_models[k]['f1'])
            
            # Fusion vs Dense
            result = compute_mcnemar_test(
                successful_models['dense_only']['predictions'],
                None,
                successful_models[best_fusion]['predictions'],
                successful_models[best_fusion]['labels']
            )
            mcnemar_results[f"{best_fusion}_vs_dense"] = result
            
            # Fusion vs KG
            result = compute_mcnemar_test(
                None,
                successful_models['kg_only']['predictions'],
                successful_models[best_fusion]['predictions'],
                successful_models[best_fusion]['labels']
            )
            mcnemar_results[f"{best_fusion}_vs_kg"] = result
        
        results['statistical_tests'] = {
            'mcnemar': mcnemar_results
        }
        
        # Print summary
        print("\nComplementarity Analysis Summary:")
        print(f"{'Metric':<30} {'Value':<10} {'Target'}")
        print("-" * 60)
        for metric, value in complementarity_analysis.items():
            if isinstance(value, float):
                target = ""
                if metric == 'gap_closure' and value >= 0.4:
                    target = "✅ ≥0.40"
                elif metric == 'both_tnr' and value >= 0.90:
                    target = "✅ ≥0.90"
                elif metric == 'kg_slice_acc' and value >= 0.85:
                    target = "✅ ≥0.85"
                print(f"  {metric:<30} {value:.3f}     {target}")
        
        # Print McNemar results
        print("\nStatistical Significance (McNemar's Test):")
        for pair, result in mcnemar_results.items():
            p = result.get('p_value', result.get('mcnemar_p_value', 1.0))
            sig = "✅" if p < 0.05 else "❌"
            print(f"  {pair:<40} p={p:.4f} {sig}")
            if 'interpretation' in result:
                print(f"    {result['interpretation']}")
    
    # Add experiment metadata
    results['experiment_metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'num_samples': num_samples,
        'num_epochs': num_epochs,
        'seed': seed,
        'calibrated_thresholds': calibrate_thresholds,
        'total_runtime_seconds': time.time() - start_time,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    }
    
    # Save results with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path('results/experiments')
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f'complementarity_results_{timestamp}.json'
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results = convert_numpy_types(results)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results summary
    print("\n📊 Results Summary:")
    print("=" * 50)
    for model_name, result in results.items():
        if result.get('status') != 'failed':
            f1 = result.get('f1', 0.0)
            acc = result.get('accuracy', 0.0)
            print(f"  {model_name}: F1 = {f1:.3f}, Acc = {acc:.3f}")
        else:
            print(f"  {model_name}: FAILED - {result.get('error', 'Unknown error')}")
    
    # Add agreement diagnostics
    print("\n🤝 Model Agreement Analysis:")
    print("=" * 50)
    if 'dense_only' in results and 'kg_only' in results:
        d_preds = np.array(results['dense_only']['predictions'])
        k_preds = np.array(results['kg_only']['predictions'])
        agreement = np.mean(d_preds == k_preds)
        print(f"  Dense–KG agreement: {agreement:.3f}")
        
        # Per-slice positive rates
        slice_names = ['dense', 'kg', 'both', 'none']
        slice_labels = np.array(results['dense_only']['slice_labels'])
        labels = np.array(results['dense_only']['labels'])
        
        for s, name in enumerate(slice_names):
            mask = slice_labels == s
            if np.sum(mask) > 0:
                pos_rate = np.mean(labels[mask])
                n_samples = np.sum(mask)
                print(f"  Slice {name}: positive rate = {pos_rate:.2f} (n={n_samples})")
    
    # Analyze complementarity
    print("\n🔍 Complementarity Analysis:")
    print("=" * 50)
    analyze_complementarity(results)
    
    # Gate diagnostics for hierarchical gating fusion
    if 'fusion_hierarchical_gating' in results:
        print("\n🚪 Gate Diagnostics:")
        print("=" * 50)
        analyze_gate_diagnostics(results)
    
    # Print complementarity analysis
    print_experiment_summary(results)
    
    # Add McNemar test for best fusion model
    best_fusion_key = None
    best_fusion_acc = 0
    for name, result in results.items():
        if name not in ['dense_only', 'kg_only'] and result.get('status') == 'success':
            if result.get('accuracy', 0) > best_fusion_acc:
                best_fusion_acc = result['accuracy']
                best_fusion_key = name
    
    if best_fusion_key and 'predictions' in results[best_fusion_key]:
        mcnemar_stats = compute_mcnemar_test(
            results['dense_only']['predictions'],
            results['kg_only']['predictions'], 
            results[best_fusion_key]['predictions'],
            results['dense_only']['labels']
        )
        print(f"\nMcNemar Test ({best_fusion_key} vs Dense):")
        print(f"  n01 (Dense wrong, Fusion right): {mcnemar_stats['mcnemar_n01']}")
        print(f"  n10 (Dense right, Fusion wrong): {mcnemar_stats['mcnemar_n10']}")
        if mcnemar_stats['mcnemar_p_value'] is not None:
            print(f"  p-value: {mcnemar_stats['mcnemar_p_value']:.4f}")
        else:
            print(f"  p-value: N/A (small sample)")
    
    print(f"\n📁 Results saved to: {results_file}")
    print("🏁 Complementarity experiments completed!")
    
    return results

def analyze_gate_diagnostics(results: Dict[str, Dict]):
    """Analyze gate behavior for hierarchical gating fusion model"""
    if 'fusion_hierarchical_gating' not in results:
        print("  No hierarchical gating fusion results found")
        return
    
    fusion_result = results['fusion_hierarchical_gating']
    if 'predictions' not in fusion_result or 'slice_labels' not in fusion_result:
        print("  Missing prediction data for gate analysis")
        return
    
    predictions = np.array(fusion_result['predictions'])
    slice_labels = np.array(fusion_result['slice_labels'])
    labels = np.array(fusion_result['labels'])
    
    slice_names = ['dense', 'kg', 'both', 'none']
    
    print("  Gate behavior analysis (based on predictions):")
    for s, name in enumerate(slice_names):
        mask = slice_labels == s
        if np.sum(mask) > 0:
            slice_preds = predictions[mask]
            slice_true = labels[mask]
            n_samples = np.sum(mask)
            pos_pred_rate = np.mean(slice_preds)
            pos_true_rate = np.mean(slice_true)
            accuracy = np.mean(slice_preds == slice_true)
            
            print(f"    {name:6s}: n={n_samples} | pred_pos={pos_pred_rate:.2f} | true_pos={pos_true_rate:.2f} | acc={accuracy:.2f}")
    
    # Overall gate effectiveness
    dense_mask = slice_labels == 0
    kg_mask = slice_labels == 1
    both_mask = slice_labels == 2
    
    if np.sum(dense_mask) > 0 and np.sum(kg_mask) > 0:
        dense_acc = np.mean(predictions[dense_mask] == labels[dense_mask])
        kg_acc = np.mean(predictions[kg_mask] == labels[kg_mask])
        print(f"  Specialization: Dense={dense_acc:.2f}, KG={kg_acc:.2f}")
        
    if np.sum(both_mask) > 0:
        both_acc = np.mean(predictions[both_mask] == labels[both_mask])
        print(f"  Fusion effectiveness on 'both' slice: {both_acc:.2f}")

def compute_mcnemar_test(dense_preds=None, kg_preds=None, fusion_preds=None, labels=None):
    """
    Compute McNemar stats (fusion vs one baseline). Provide either dense_preds or kg_preds.
    
    Args:
        dense_preds: Predictions from the dense model (optional)
        kg_preds: Predictions from the KG model (optional)
        fusion_preds: Predictions from the fusion model (required)
        labels: Ground truth labels (required)
        
    Returns:
        Dictionary with test statistics including p-value and interpretation
    """
    if fusion_preds is None or labels is None:
        raise ValueError("fusion_preds and labels are required.")
    baseline_preds = dense_preds if dense_preds is not None else kg_preds
    if baseline_preds is None:
        raise ValueError("Provide either dense_preds or kg_preds as baseline.")

    # numpy-ize
    baseline_preds = np.asarray(baseline_preds).astype(int).ravel()
    fusion_preds = np.asarray(fusion_preds).astype(int).ravel()
    labels = np.asarray(labels).astype(int).ravel()

    # correctness
    b_ok = (baseline_preds == labels)
    f_ok = (fusion_preds == labels)

    # McNemar counts
    n01 = int(np.sum((~b_ok) & f_ok))   # baseline wrong, fusion right
    n10 = int(np.sum(b_ok & (~f_ok)))   # baseline right, fusion wrong
    n = n01 + n10

    if n == 0:
        p = 1.0
    else:
        # exact two-sided p = 2 * sum_{i=0}^{min(n01,n10)} C(n,i) / 2^n
        from math import comb
        m = min(n01, n10)
        p_tail = sum(comb(n, i) for i in range(0, m + 1)) * (0.5 ** n)
        p = float(min(1.0, 2 * p_tail))

    return {
        "mcnemar_n01": n01,
        "mcnemar_n10": n10,
        "mcnemar_p_value": p,
        "p_value": p,  # alias for convenience
        "interpretation": f"n01={n01} (baseline wrong, fusion right), n10={n10} (baseline right, fusion wrong)"
    }

def analyze_complementarity(results: Dict[str, Dict]) -> Dict[str, float]:
    """Analyze complementarity between Dense and KG models (array-safe) and include McNemar."""
    import numpy as np

    def _present_ok(key: str) -> bool:
        """Check that a model exists, didn't fail, and has non-empty preds/labels (array-safe)."""
        if key not in results:
            return False
        r = results.get(key, {})
        if r.get('status') == 'failed':
            return False
        preds = r.get('predictions', None)
        labels = r.get('labels', None)
        if preds is None or labels is None:
            return False
        try:
            return (len(preds) > 0) and (len(labels) > 0)
        except Exception:
            return False

    if not (_present_ok('dense_only') and _present_ok('kg_only')):
        print("  Skipping complementarity analysis (dense_only/kg_only missing or failed)")
        return {}

    d = np.asarray(results['dense_only']['predictions']).astype(int).ravel()
    k = np.asarray(results['kg_only']['predictions']).astype(int).ravel()
    y = np.asarray(results['dense_only']['labels']).astype(int).ravel()

    d_ok, k_ok = (d == y), (k == y)
    dense_acc, kg_acc = d_ok.mean(), k_ok.mean()
    oracle_acc = (d_ok | k_ok).mean()
    best_single = max(dense_acc, kg_acc)
    agree = (d == k).mean()

    gap_closures, CIs, f_accs = {}, {}, []
    best_fusion_acc = 0.0
    best_fusion_key = None
    best_fusion_preds = None
    
    for name, r in results.items():
        if name in ('dense_only', 'kg_only') or r.get('status') == 'failed':
            continue
        if 'predictions' not in r or 'labels' not in r:
            continue
        f = np.asarray(r['predictions']).astype(int).ravel()
        f_ok = (f == y)
        f_acc = f_ok.mean()
        f_accs.append(f_acc)
        
        if f_acc >= best_fusion_acc:
            best_fusion_acc = f_acc
            best_fusion_key = name
            best_fusion_preds = f
        denom = max(oracle_acc - best_single, 1e-9)
        # keep gap_closure in [0,1] for reporting
        gap_closures[name] = max(0.0, min(1.0, (f_acc - best_single) / denom))
        CIs[name] = np.mean(f_ok & (~d_ok) & (~k_ok))

    best_gap = max(gap_closures.values()) if gap_closures else 0.0
    avg_gap = float(np.mean(list(gap_closures.values()))) if gap_closures else 0.0
    
    # McNemar (exact, two-sided) vs best single
    if best_fusion_preds is None:
        mcnemar_p = 1.0
        n01 = n10 = 0
    else:
        best_single_preds = d if dense_acc >= kg_acc else k
        b_ok = (best_single_preds == y)
        f_ok = (best_fusion_preds == y)
        n01 = int(np.sum(b_ok & (~f_ok)))  # best correct, fusion wrong
        n10 = int(np.sum((~b_ok) & f_ok))  # best wrong, fusion correct
        n = n01 + n10
        if n > 0:
            from math import comb
            tail = sum(comb(n, i) for i in range(min(n01, n10), -1, -1)) * (0.5 ** n)
            mcnemar_p = float(min(1.0, 2 * tail))
        else:
            mcnemar_p = 1.0

    return {
        'dense_accuracy': float(dense_acc),
        'kg_accuracy': float(kg_acc),
        'oracle_accuracy': float(oracle_acc),
        'best_single_accuracy': float(best_single),
        'best_gap_closure': float(best_gap),
        'gap_closure': float(avg_gap),
        'complementarity_index': float(np.mean(list(CIs.values())) if CIs else 0.0),
        'dense_kg_agreement': float(agree),
        'best_fusion_accuracy': float(best_fusion_acc),
        'avg_fusion_accuracy': float(np.mean(f_accs) if f_accs else 0.0),
        'fusion_accuracy': float(best_fusion_acc),
        'mcnemar_p_value': float(mcnemar_p),
        'mcnemar_n01': n01,
        'mcnemar_n10': n10,
        'best_fusion_key': best_fusion_key or "",
    }

def compute_advanced_metrics(dense_preds, kg_preds, fusion_preds, labels):
    """
    Compute advanced evaluation metrics for complementarity analysis
    
    Args:
        dense_preds: Dense model predictions [N]
        kg_preds: KG model predictions [N] 
        fusion_preds: Fusion model predictions [N]
        labels: Ground truth labels [N]
    
    Returns:
        Dictionary with advanced metrics
    """
    # Convert to numpy arrays and ensure integer predictions
    d = np.asarray(dense_preds).astype(int)
    k = np.asarray(kg_preds).astype(int)
    f = np.asarray(fusion_preds).astype(int)
    y = np.asarray(labels).astype(int)
    
    # Accuracies
    d_ok, k_ok, f_ok = (d == y), (k == y), (f == y)
    d_acc, k_acc, f_acc = d_ok.mean(), k_ok.mean(), f_ok.mean()
    
    # Oracle accuracy = max of (dense, kg) per item
    oracle_ok = d_ok | k_ok
    oracle_acc = oracle_ok.mean()
    best_single = max(d_acc, k_acc)
    
    # Gap closure
    denom = max(oracle_acc - best_single, 1e-9)
    gap_closure = (f_acc - best_single) / denom if denom > 1e-9 else 0.0
    
    # Complementarity Index (fusion wins when both singles fail)
    ci = np.mean(f_ok & (~d_ok) & (~k_ok))
    
    # Dense-KG agreement
    agree = np.mean(d == k)
    
    # McNemar test (fusion vs best single)
    best = d if d_acc >= k_acc else k
    b_ok = (best == y)
    n01 = np.sum(b_ok & (~f_ok))  # best correct, fusion wrong
    n10 = np.sum((~b_ok) & f_ok)  # best wrong, fusion correct
    
    # Exact binomial test for McNemar
    if n01 + n10 > 0:
        n = n01 + n10
        p = sum(comb(n, i) for i in range(n10, n+1)) * (0.5 ** n)
        mcnemar_p = min(1.0, 2*p)  # two-sided
    else:
        mcnemar_p = 1.0
    
    return dict(
        dense_accuracy=float(d_acc),
        kg_accuracy=float(k_acc),
        fusion_accuracy=float(f_acc),
        oracle_accuracy=float(oracle_acc),
        gap_closure=float(gap_closure),
        complementarity_index=float(ci),
        dense_kg_agreement=float(agree),
        mcnemar_p_value=float(mcnemar_p),
        mcnemar_n01=int(n01),
        mcnemar_n10=int(n10)
    )

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
            key=lambda x: x[1].get('f1', 0.0),
            reverse=True
        )
        
        print("\nModel Performance Ranking:")
        for i, (model_key, model_results) in enumerate(sorted_models, 1):
            f1 = model_results.get('f1', 0.0)
            accuracy = model_results.get('accuracy', 0.0)
            params = model_results.get('parameters', 0) / 1e6
            
            print(f"  {i}. {model_results.get('model_name', model_key):25}: "
                  f"F1={f1:.3f}, "
                  f"Acc={accuracy:.3f}, "
                  f"Params={params:.1f}M")
        
        # Complementarity analysis
        if 'complementarity_analysis' in results:
            ca = results['complementarity_analysis']
            print(f"\n Complementarity Analysis (advanced):")
            print(f"  Dense acc: {ca.get('dense_accuracy', 0):.3f} | KG acc: {ca.get('kg_accuracy', 0):.3f} | Best fusion acc: {ca.get('best_fusion_accuracy', 0):.3f}")
            print(f"  Oracle accuracy: {ca.get('oracle_accuracy', 0):.3f}")
            print(f"  Gap closure: {ca.get('gap_closure', 0):.3f}  (target ≥ 0.30)")
            print(f"  Complementarity Index: {ca.get('complementarity_index', 0):.3f}  (target ≥ 0.05)")
            print(f"  Dense–KG agreement: {ca.get('dense_kg_agreement', 0):.3f}")
            print(f"  McNemar p-value: {ca.get('mcnemar_p_value', 1.0):.4f}  (n01={ca.get('mcnemar_n01', 0)}, n10={ca.get('mcnemar_n10', 0)})")
            print(f"  Best fusion acc: {ca.get('best_fusion_accuracy', 0):.3f} | Avg fusion acc: {ca.get('avg_fusion_accuracy', 0):.3f}")
            
            # Success criteria
            gap_closure = ca.get('gap_closure', 0)
            ci = ca.get('complementarity_index', 0)
            mcnemar_p = ca.get('mcnemar_p_value', 1.0)
            
            if gap_closure >= 0.30:
                print("  Gap closure target achieved!")
            if ci >= 0.05:
                print("  Complementarity Index target achieved!")
            if mcnemar_p < 0.01:
                print("  Statistical significance achieved!")
            if gap_closure < 0.30 or ci < 0.05:
                print("  Targets not met. Consider different fusion strategies.")
    
    else:
        print(" No models completed successfully")

def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description='Run complementarity experiments')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples per dataset')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    
    args = parser.parse_args()
    
    results = run_complementarity_experiments(
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        seed=42
    )
    
    return results

if __name__ == "__main__":
    main()