#!/usr/bin/env python3
"""
Baseline Model Architectures for KG + Dense Vector Complementarity Research
Implements fusion models for QA and IR tasks on M1 MacBook
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, RGCNConv
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Any
import math

class DenseVectorEncoder(nn.Module):
    """Dense text encoder using pre-trained language model"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", 
                 hidden_dim: int = 768, freeze_lm: bool = False, pooling: str = "cls"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling  # "cls" | "mean" | "attn"
        
        if freeze_lm:
            # Freeze language model parameters except last layer for adaptation
            for name, param in self.encoder.named_parameters():
                if "layer.5" not in name and "embeddings" not in name:  # Keep last layer + embeddings unfrozen
                    param.requires_grad = False
        
        # Project to desired hidden dimension
        encoder_dim = self.encoder.config.hidden_size
        self.projection = nn.Linear(encoder_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Attention pooling head if needed
        if pooling == "attn":
            self.attn_pool = nn.Linear(encoder_dim, 1)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_sequence: If True, return full token sequence for cross-attention
        Returns:
            Dense embeddings [batch_size, hidden_dim] or [batch_size, seq_len, hidden_dim]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        if return_sequence:
            # Return full sequence for cross-attention
            return self.dropout(self.projection(sequence_output))
        
        # Pooling strategies
        if self.pooling == "cls":
            # Use CLS token (first token) - preserves bigram order info
            pooled_output = sequence_output[:, 0]  # [batch_size, hidden_size]
        elif self.pooling == "attn":
            # Attention-weighted pooling
            scores = self.attn_pool(sequence_output).squeeze(-1)  # [batch_size, seq_len]
            scores = scores.masked_fill(attention_mask == 0, -1e9)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]
            pooled_output = (sequence_output * weights).sum(1)  # [batch_size, hidden_size]
        else:  # "mean"
            # Masked mean pooling (ignore padding tokens)
            mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
            masked_embeddings = sequence_output * mask
            pooled_output = masked_embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        
        # Apply projection and dropout
        return self.dropout(self.projection(pooled_output))

class KnowledgeGraphEncoder(nn.Module):
    """Knowledge graph encoder using Graph Neural Networks"""
    
    def __init__(self, num_entities, num_relations, hidden_dim=768, gnn_type="gcn", num_layers=2, num_bases=16, dropout=0.1):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, hidden_dim)
        
        # Relation embeddings
        self.relation_embeddings = nn.Embedding(num_relations, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == "gcn":
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == "gat":
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout))
            elif gnn_type == "sage":
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
            elif gnn_type == "rgcn":
                self.gnn_layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases))
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, entity_ids: torch.Tensor, edge_index: torch.Tensor, 
                edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            entity_ids: Batch entity IDs to gather [batch_size]
            edge_index: Full graph edge structure [2, num_edges]
            edge_type: Edge relation types [num_edges] (required for RGCN)
        Returns:
            Entity embeddings for batch [batch_size, hidden_dim]
        """
        if self.gnn_type == "rgcn" and edge_type is None:
            raise ValueError("edge_type is required when gnn_type='rgcn'")
        # Full-graph compute then gather approach
        x = self.entity_embeddings.weight  # [num_entities, hidden_dim]
        
        # Apply GNN layers on full graph
        for i, gnn_layer in enumerate(self.gnn_layers):
            x_residual = x
            if self.gnn_type == "rgcn":
                x = gnn_layer(x, edge_index, edge_type)
            else:
                x = gnn_layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Residual connection
            if x.shape == x_residual.shape:
                x = x + x_residual
            
            x = self.layer_norm(x)
        
        # Gather specific entities for batch
        return x[entity_ids]

class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion mechanism between KG and Dense vectors"""
    
    def __init__(self, hidden_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Cross-attention layers
        self.kg_to_dense_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1, batch_first=True)
        self.dense_to_kg_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1, batch_first=True)
        
        # Feed-forward networks
        self.kg_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.dense_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, kg_embeddings: torch.Tensor, dense_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            kg_embeddings: KG embeddings [batch_size, hidden_dim] or [batch_size, seq_len, hidden_dim]
            dense_embeddings: Dense embeddings [batch_size, hidden_dim] or [batch_size, seq_len, hidden_dim]
        Returns:
            Fused KG and Dense embeddings
        """
        # Handle both pooled and sequence inputs
        if kg_embeddings.dim() == 2:
            kg_embeddings = kg_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        if dense_embeddings.dim() == 2:
            dense_embeddings = dense_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Cross-attention with batch_first=True (no transpose needed)
        kg_attended, _ = self.kg_to_dense_attention(query=kg_embeddings, key=dense_embeddings, value=dense_embeddings)
        kg_attended = self.layer_norm(kg_attended + kg_embeddings)
        kg_attended = self.layer_norm(kg_attended + self.kg_ffn(kg_attended))
        
        dense_attended, _ = self.dense_to_kg_attention(query=dense_embeddings, key=kg_embeddings, value=kg_embeddings)
        dense_attended = self.layer_norm(dense_attended + dense_embeddings)
        dense_attended = self.layer_norm(dense_attended + self.dense_ffn(dense_attended))
        
        # Return first token (CLS) if sequence input, otherwise squeeze
        if kg_attended.size(1) > 1:
            kg_fused = kg_attended[:, 0, :]  # CLS token
            dense_fused = dense_attended[:, 0, :]  # CLS token
        else:
            kg_fused = kg_attended.squeeze(1)
            dense_fused = dense_attended.squeeze(1)
        
        return kg_fused, dense_fused

class HierarchicalGatingFusion(nn.Module):
    """Hierarchical gating mechanism for adaptive fusion with supervision support"""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_temp = 0.7  # Temperature for sharper softmax
        
        # Gating networks
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Auxiliary heads for oracle distillation
        self.dense_aux_head = nn.Linear(hidden_dim, 1)
        self.kg_aux_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, kg_embeddings: torch.Tensor, dense_embeddings: torch.Tensor, 
                return_gates: bool = False, return_aux_logits: bool = False) -> torch.Tensor:
        """
        Args:
            kg_embeddings: KG embeddings [batch_size, hidden_dim]
            dense_embeddings: Dense embeddings [batch_size, hidden_dim]
            return_gates: Return gate values for supervision
            return_aux_logits: Return auxiliary logits for oracle distillation
        Returns:
            Fused embeddings and optionally gates/aux_logits
        """
        # Compute gates with competitive softmax (not sigmoid+renorm)
        combined = torch.cat([kg_embeddings, dense_embeddings], dim=-1)
        gate_logits = self.gate_head(combined)        # Compute gates with temperature-scaled softmax for sharpness
        gates = F.softmax(gate_logits / self.gate_temp, dim=-1)  # Competitive gating
        
        # Apply gates
        kg_gated = gates[:, 0:1] * kg_embeddings
        dense_gated = gates[:, 1:2] * dense_embeddings
        
        # Fuse
        fused = torch.cat([kg_gated, dense_gated], dim=-1)
        fused_output = self.fusion_layer(fused)
        
        if return_gates or return_aux_logits:
            outputs = [fused_output]
            if return_gates:
                outputs.extend([gates[:, 0:1], gates[:, 1:2]])
            if return_aux_logits:
                kg_aux = self.kg_aux_head(kg_embeddings)
                dense_aux = self.dense_aux_head(dense_embeddings)
                outputs.extend([kg_aux, dense_aux])
            return tuple(outputs)
        
        return fused_output

class TensorBilinearFusion(nn.Module):
    """Tensor bilinear fusion for complex KG-Dense interactions"""
    
    def __init__(self, hidden_dim: int = 768, fusion_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        
        # Bilinear tensor for interaction modeling
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, fusion_dim)
        
        # Projection layers
        self.kg_proj = nn.Linear(hidden_dim, fusion_dim)
        self.dense_proj = nn.Linear(hidden_dim, fusion_dim)
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 3, hidden_dim),  # bilinear + kg_proj + dense_proj
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, kg_embeddings: torch.Tensor, dense_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kg_embeddings: KG embeddings [batch_size, hidden_dim]
            dense_embeddings: Dense embeddings [batch_size, hidden_dim]
        Returns:
            Fused embeddings [batch_size, hidden_dim]
        """
        # Bilinear interaction
        bilinear_interaction = self.bilinear(kg_embeddings, dense_embeddings)
        
        # Individual projections
        kg_proj = self.kg_proj(kg_embeddings)
        dense_proj = self.dense_proj(dense_embeddings)
        
        # Concatenate all interactions
        combined = torch.cat([bilinear_interaction, kg_proj, dense_proj], dim=-1)
        
        # Final fusion
        fused = self.fusion_layer(combined)
        fused = self.layer_norm(fused)
        
        return fused

class ContrastiveFusion(nn.Module):
    """Contrastive learning-based fusion mechanism"""
    
    def __init__(self, hidden_dim: int = 768, temperature: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # Projection heads for contrastive learning
        self.kg_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.dense_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Alignment and uniformity components
        self.alignment_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, kg_embeddings: torch.Tensor, dense_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kg_embeddings: KG embeddings [batch_size, hidden_dim]
            dense_embeddings: Dense embeddings [batch_size, hidden_dim]
        Returns:
            Fused embeddings [batch_size, hidden_dim]
        """
        # Project to contrastive space
        kg_proj = self.kg_projector(kg_embeddings)
        dense_proj = self.dense_projector(dense_embeddings)
        
        # Normalize projections
        kg_proj = F.normalize(kg_proj, dim=-1)
        dense_proj = F.normalize(dense_proj, dim=-1)
        
        # Compute alignment scores
        alignment_scores = torch.sum(kg_proj * dense_proj, dim=-1, keepdim=True)
        
        # Weighted combination based on alignment
        alignment_weights = torch.sigmoid(alignment_scores)
        
        # Combine original embeddings
        combined = torch.cat([
            alignment_weights * kg_embeddings,
            (1 - alignment_weights) * dense_embeddings
        ], dim=-1)
        
        # Alignment-aware fusion
        aligned = self.alignment_layer(combined)
        fused = self.fusion_layer(aligned)
        fused = self.layer_norm(fused)
        
        return fused
    
    def contrastive_loss(self, kg_embeddings: torch.Tensor, dense_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for training"""
        kg_proj = F.normalize(self.kg_projector(kg_embeddings), dim=-1)
        dense_proj = F.normalize(self.dense_projector(dense_embeddings), dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(kg_proj, dense_proj.T) / self.temperature
        
        # Positive pairs are on the diagonal
        batch_size = kg_embeddings.size(0)
        labels = torch.arange(batch_size, device=kg_embeddings.device)
        
        # Symmetric contrastive loss
        loss_kg_to_dense = F.cross_entropy(similarity, labels)
        loss_dense_to_kg = F.cross_entropy(similarity.T, labels)
        
        return (loss_kg_to_dense + loss_dense_to_kg) / 2

class KGDenseFusionModel(nn.Module):
    """Main fusion model combining KG and Dense representations"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Encoders
        self.dense_encoder = DenseVectorEncoder(
            model_name=config.get('dense_model', 'distilbert-base-uncased'),
            hidden_dim=config.get('hidden_dim', 768),
            freeze_lm=config.get('freeze_lm', True),
            pooling=config.get('pooling', 'mean')
        )
        self.kg_encoder = KnowledgeGraphEncoder(
            num_entities=config['num_entities'],
            num_relations=config['num_relations'],
            hidden_dim=config['hidden_dim'],
            num_layers=config.get('gnn_layers', 2),
            num_bases=config.get('num_bases', 16),
            dropout=config.get('dropout', 0.1)
        )
        
        # Fusion mechanisms
        fusion_type = config.get('fusion_type', 'cross_attention')
        if fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(
                hidden_dim=config.get('hidden_dim', 768),
                num_heads=config.get('num_heads', 8)
            )
        elif fusion_type == 'hierarchical_gating':
            self.fusion = HierarchicalGatingFusion(
                hidden_dim=config.get('hidden_dim', 768)
            )
        elif fusion_type == 'tensor_bilinear':
            self.fusion = TensorBilinearFusion(
                hidden_dim=config.get('hidden_dim', 768),
                fusion_dim=config.get('fusion_dim', 256)
            )
        elif fusion_type == 'contrastive':
            self.fusion = ContrastiveFusion(
                hidden_dim=config.get('hidden_dim', 768),
                temperature=config.get('temperature', 0.1)
            )
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        
        self.fusion_type = fusion_type
        
        # Learned interaction residual for AND patterns
        self.interaction_head = nn.Sequential(
            nn.Linear(config.get('hidden_dim', 768), config.get('hidden_dim', 768)),
            nn.ReLU(),
            nn.Linear(config.get('hidden_dim', 768), config.get('hidden_dim', 768))
        )
        
        # Task-specific heads
        self.qa_head = nn.Sequential(
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(config.get('hidden_dim', 768), 1)
        )
        self.ir_head = nn.Sequential(
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(config.get('hidden_dim', 768) * 2, 1)
        )
        
    def forward(self, batch: Dict[str, torch.Tensor], task: str = 'qa', return_aux_outputs: bool = False) -> torch.Tensor:
        """
        Args:
            batch: Input batch containing text and KG data
            task: Task type ('qa' or 'ir')
            return_aux_outputs: Return auxiliary outputs for supervision
        Returns:
            Task-specific predictions and optionally auxiliary outputs
        """
        # Get embeddings
        dense_embeddings = self.dense_encoder(batch['input_ids'], batch['attention_mask'])
        kg_embeddings = self.kg_encoder(batch['entity_ids'], batch['edge_index'], batch.get('edge_type'))
        
        # Fusion with auxiliary outputs for hierarchical gating
        if self.fusion_type == 'cross_attention':
            # 1) Get token sequence from the text encoder
            dense_seq = self.dense_encoder(batch['input_ids'], batch['attention_mask'], return_sequence=True)  # [B,T,H]

            # 2) Tile the KG vector across tokens (MPS-safe: repeat, not expand)
            kg_vec = self.kg_encoder(batch['entity_ids'], batch['edge_index'], batch.get('edge_type'))        # [B,H]
            T = dense_seq.size(1)
            kg_seq = kg_vec.unsqueeze(1).repeat(1, T, 1).contiguous()                                          # [B,T,H]

            # 3) Cross-attend both ways (batch_first=True)
            kg_fused, dense_fused = self.fusion(kg_seq, dense_seq)  # token-level X-attn
            fused_embeddings = 0.5 * (kg_fused + dense_fused)  # Average both streams
            aux_outputs = None
        elif self.fusion_type == 'hierarchical_gating' and return_aux_outputs:
            # Get gates and auxiliary logits for supervision
            fused_embeddings, kg_gate, dense_gate, kg_aux, dense_aux = self.fusion(
                kg_embeddings, dense_embeddings, return_gates=True, return_aux_logits=True
            )
            aux_outputs = {
                'kg_gate': kg_gate,
                'dense_gate': dense_gate,
                'kg_aux_logits': kg_aux,
                'dense_aux_logits': dense_aux
            }
        else:
            fused_embeddings = self.fusion(kg_embeddings, dense_embeddings)
            aux_outputs = None
        
        # Add learned interaction residual (kg * dense) for AND patterns
        interaction = self.interaction_head(kg_embeddings * dense_embeddings)
        fused_embeddings = fused_embeddings + interaction
        
        # Task-specific heads
        if task == 'qa':
            logits = self.qa_head(fused_embeddings)
        elif task == 'ir':
            # For IR, we need passage embeddings too
            passage_embeddings = batch.get('passage_embeddings', fused_embeddings)
            combined = torch.cat([fused_embeddings, passage_embeddings], dim=-1)
            logits = self.ir_head(combined)
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        if return_aux_outputs and aux_outputs is not None:
            return logits, aux_outputs
        return logits

class BaselineModels:
    """Factory class for creating baseline models"""
    
    @staticmethod
    def create_dense_only_model(config: Dict[str, Any]) -> nn.Module:
        """Create dense-only baseline model"""
        class DenseOnlyModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.encoder = DenseVectorEncoder(
                    model_name=config.get('dense_model', 'distilbert-base-uncased'),
                    hidden_dim=config.get('hidden_dim', 768),
                    freeze_lm=config.get('freeze_lm', False),
                    pooling=config.get('pooling', 'mean')
                )
                self.qa_head = nn.Linear(config.get('hidden_dim', 768), 1)
                self.ir_head = nn.Linear(config.get('hidden_dim', 768) * 2, 1)
            
            def forward(self, batch, task='qa'):
                embeddings = self.encoder(batch['input_ids'], batch['attention_mask'])
                if task == 'qa':
                    return self.qa_head(embeddings)
                else:
                    combined = torch.cat([embeddings, embeddings], dim=-1)
                    return self.ir_head(combined)
        
        return DenseOnlyModel(config)
    
    @staticmethod
    def create_kg_only_model(config: Dict[str, Any]) -> nn.Module:
        """Create KG-only baseline model"""
        class KGOnlyModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.encoder = KnowledgeGraphEncoder(
                    num_entities=config.get('num_entities', 100000),
                    num_relations=config.get('num_relations', 1000),
                    hidden_dim=config.get('hidden_dim', 768),
                    gnn_type=config.get('gnn_type', 'gcn'),
                    num_layers=config.get('gnn_layers', 2),
                    num_bases=config.get('num_bases', 16)
                )
                self.qa_head = nn.Linear(config.get('hidden_dim', 768), 1)
                self.ir_head = nn.Linear(config.get('hidden_dim', 768) * 2, 1)
            
            def forward(self, batch, task='qa'):
                embeddings = self.encoder(batch['entity_ids'], batch['edge_index'], batch.get('edge_type'))
                if task == 'qa':
                    return self.qa_head(embeddings)
                else:
                    combined = torch.cat([embeddings, embeddings], dim=-1)
                    return self.ir_head(combined)
        
        return KGOnlyModel(config)
    
    @staticmethod
    def create_fusion_model(config: Dict[str, Any]) -> KGDenseFusionModel:
        """Create KG+Dense fusion model"""
        return KGDenseFusionModel(config)

# Model configurations for different experiments
MODEL_CONFIGS = {
    'm1_optimized': {
        'hidden_dim': 768,
        'num_entities': 500,
        'num_relations': 20,
        'num_bases': 16,
        'gnn_layers': 3,  # Increased from 2
        'dropout': 0.05,  # Added dropout
        'batch_size': 32,
        'learning_rate': 1e-4,
        'max_seq_length': 128
    },
    'lightweight': {
        'dense_model': 'distilbert-base-uncased',
        'hidden_dim': 384,
        'gnn_type': 'gcn',
        'gnn_layers': 2,
        'num_heads': 6,
        'dropout': 0.1,
        'fusion_dim': 256,
        'temperature': 0.1,
        'num_entities': 50000,
        'num_relations': 500,
        'freeze_lm': True
    },
    'standard': {
        'dense_model': 'bert-base-uncased',
        'hidden_dim': 768,
        'gnn_type': 'gat',
        'gnn_layers': 3,
        'num_heads': 12,
        'fusion_type': 'hierarchical_gating',
        'fusion_dim': 384,
        'temperature': 0.1,
        'num_entities': 100000,
        'num_relations': 1000
    },
    'large': {
        'dense_model': 'bert-large-uncased',
        'hidden_dim': 1024,
        'gnn_type': 'sage',
        'gnn_layers': 4,
        'num_heads': 16,
        'fusion_type': 'tensor_bilinear',
        'fusion_dim': 512,
        'temperature': 0.05,
        'num_entities': 200000,
        'num_relations': 2000
    }
}

def get_model_config(size: str = 'lightweight') -> Dict[str, Any]:
    """Get model configuration by size"""
    if size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {size}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[size].copy()

def create_model(model_type: str, size: str = 'lightweight') -> nn.Module:
    """
    Create model by type and size
    
    Args:
        model_type: 'dense_only', 'kg_only', or 'fusion'
        size: 'lightweight', 'standard', or 'large'
    """
    config = get_model_config(size)
    
    if model_type == 'dense_only':
        return BaselineModels.create_dense_only_model(config)
    elif model_type == 'kg_only':
        return BaselineModels.create_kg_only_model(config)
    elif model_type == 'fusion':
        return BaselineModels.create_fusion_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Test model creation
    print("Testing model architectures...")
    
    # Create lightweight models for M1 testing
    dense_model = create_model('dense_only', 'lightweight')
    kg_model = create_model('kg_only', 'lightweight')
    fusion_model = create_model('fusion', 'lightweight')
    
    print(f"Dense-only model parameters: {sum(p.numel() for p in dense_model.parameters()):,}")
    print(f"KG-only model parameters: {sum(p.numel() for p in kg_model.parameters()):,}")
    print(f"Fusion model parameters: {sum(p.numel() for p in fusion_model.parameters()):,}")
    
    print("✓ Model architectures created successfully!")