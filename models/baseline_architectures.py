#!/usr/bin/env python3
"""
Baseline Model Architectures for KG + Dense Vector Complementarity Research
Implements fusion models for QA and IR tasks on M1 MacBook
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Any
import math

class DenseVectorEncoder(nn.Module):
    """Dense vector encoder using pre-trained language models"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", hidden_dim: int = 768):
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        
        # Use lightweight models for M1 efficiency
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Projection layer to standardize dimensions
        self.projection = nn.Linear(self.encoder.config.hidden_size, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        Returns:
            Dense embeddings [batch_size, hidden_dim]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use CLS token or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            # Mean pooling over sequence length
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        embeddings = self.projection(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class KnowledgeGraphEncoder(nn.Module):
    """Knowledge graph encoder using Graph Neural Networks"""
    
    def __init__(self, num_entities: int, num_relations: int, hidden_dim: int = 768, 
                 gnn_type: str = "gcn", num_layers: int = 2):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        
        # Entity and relation embeddings
        self.entity_embeddings = nn.Embedding(num_entities, hidden_dim)
        self.relation_embeddings = nn.Embedding(num_relations, hidden_dim)
        
        # Graph neural network layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == "gcn":
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == "gat":
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.1))
            elif gnn_type == "sage":
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, entity_ids: torch.Tensor, edge_index: torch.Tensor, 
                edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            entity_ids: Entity IDs [batch_size] or [num_nodes]
            edge_index: Graph edges [2, num_edges]
            edge_type: Edge types [num_edges] (optional)
        Returns:
            KG embeddings [batch_size, hidden_dim] or [num_nodes, hidden_dim]
        """
        # Get entity embeddings
        if entity_ids.dim() == 1 and len(entity_ids) == edge_index.max().item() + 1:
            # Full graph mode
            x = self.entity_embeddings(entity_ids)
        else:
            # Batch mode - get embeddings for specific entities
            x = self.entity_embeddings(entity_ids)
            if x.dim() == 1:
                x = x.unsqueeze(0)
        
        # Apply GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            x_residual = x
            x = gnn_layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Residual connection
            if x.shape == x_residual.shape:
                x = x + x_residual
            
            x = self.layer_norm(x)
        
        return x

class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion mechanism between KG and Dense vectors"""
    
    def __init__(self, hidden_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Cross-attention layers
        self.kg_to_dense_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1)
        self.dense_to_kg_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1)
        
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
            kg_embeddings: KG embeddings [batch_size, hidden_dim]
            dense_embeddings: Dense embeddings [batch_size, hidden_dim]
        Returns:
            Fused KG and Dense embeddings
        """
        # Ensure proper dimensions for attention
        if kg_embeddings.dim() == 2:
            kg_embeddings = kg_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        if dense_embeddings.dim() == 2:
            dense_embeddings = dense_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Transpose for attention: [seq_len, batch_size, hidden_dim]
        kg_emb = kg_embeddings.transpose(0, 1)
        dense_emb = dense_embeddings.transpose(0, 1)
        
        # Cross-attention: KG attends to Dense
        kg_attended, _ = self.kg_to_dense_attention(kg_emb, dense_emb, dense_emb)
        kg_attended = self.layer_norm(kg_attended + kg_emb)
        kg_attended = self.layer_norm(kg_attended + self.kg_ffn(kg_attended))
        
        # Cross-attention: Dense attends to KG
        dense_attended, _ = self.dense_to_kg_attention(dense_emb, kg_emb, kg_emb)
        dense_attended = self.layer_norm(dense_attended + dense_emb)
        dense_attended = self.layer_norm(dense_attended + self.dense_ffn(dense_attended))
        
        # Transpose back and squeeze
        kg_fused = kg_attended.transpose(0, 1).squeeze(1)
        dense_fused = dense_attended.transpose(0, 1).squeeze(1)
        
        return kg_fused, dense_fused

class HierarchicalGatingFusion(nn.Module):
    """Hierarchical gating mechanism for adaptive fusion"""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Gating networks
        self.kg_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.dense_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, kg_embeddings: torch.Tensor, dense_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kg_embeddings: KG embeddings [batch_size, hidden_dim]
            dense_embeddings: Dense embeddings [batch_size, hidden_dim]
        Returns:
            Fused embeddings [batch_size, hidden_dim]
        """
        # Concatenate for gating
        combined = torch.cat([kg_embeddings, dense_embeddings], dim=-1)
        
        # Compute gates
        kg_gate = self.kg_gate(combined)
        dense_gate = self.dense_gate(combined)
        
        # Normalize gates
        gate_sum = kg_gate + dense_gate + 1e-8
        kg_gate = kg_gate / gate_sum
        dense_gate = dense_gate / gate_sum
        
        # Apply gates
        gated_kg = kg_gate * kg_embeddings
        gated_dense = dense_gate * dense_embeddings
        
        # Fuse
        fused = torch.cat([gated_kg, gated_dense], dim=-1)
        fused = self.fusion_layer(fused)
        
        return fused

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
            hidden_dim=config.get('hidden_dim', 768)
        )
        
        self.kg_encoder = KnowledgeGraphEncoder(
            num_entities=config.get('num_entities', 100000),
            num_relations=config.get('num_relations', 1000),
            hidden_dim=config.get('hidden_dim', 768),
            gnn_type=config.get('gnn_type', 'gcn'),
            num_layers=config.get('gnn_layers', 2)
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
        
        # Task-specific heads
        self.qa_head = nn.Sequential(
            nn.Linear(config.get('hidden_dim', 768), config.get('hidden_dim', 768)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.get('hidden_dim', 768), 1)  # Binary classification for answer span
        )
        
        self.ir_head = nn.Sequential(
            nn.Linear(config.get('hidden_dim', 768) * 2, config.get('hidden_dim', 768)),  # Query + Passage
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.get('hidden_dim', 768), 1)  # Relevance score
        )
        
    def forward(self, batch: Dict[str, torch.Tensor], task: str = 'qa') -> torch.Tensor:
        """
        Args:
            batch: Input batch containing text and KG data
            task: Task type ('qa' or 'ir')
        Returns:
            Task-specific predictions
        """
        # Encode dense representations
        dense_embeddings = self.dense_encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # Encode KG representations
        kg_embeddings = self.kg_encoder(
            entity_ids=batch['entity_ids'],
            edge_index=batch['edge_index'],
            edge_type=batch.get('edge_type')
        )
        
        # Fusion
        if self.fusion_type == 'cross_attention':
            kg_fused, dense_fused = self.fusion(kg_embeddings, dense_embeddings)
            # Combine fused representations
            fused_embeddings = (kg_fused + dense_fused) / 2
        elif self.fusion_type in ['hierarchical_gating', 'tensor_bilinear', 'contrastive']:
            fused_embeddings = self.fusion(kg_embeddings, dense_embeddings)
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
        
        # Task-specific prediction
        if task == 'qa':
            return self.qa_head(fused_embeddings)
        elif task == 'ir':
            # For IR, we need query and passage embeddings
            if 'passage_embeddings' in batch:
                combined = torch.cat([fused_embeddings, batch['passage_embeddings']], dim=-1)
            else:
                # Use the same embeddings for both query and passage
                combined = torch.cat([fused_embeddings, fused_embeddings], dim=-1)
            return self.ir_head(combined)
        else:
            raise ValueError(f"Unsupported task: {task}")

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
                    hidden_dim=config.get('hidden_dim', 768)
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
                    gnn_type=config.get('gnn_type', 'gcn')
                )
                self.qa_head = nn.Linear(config.get('hidden_dim', 768), 1)
                self.ir_head = nn.Linear(config.get('hidden_dim', 768) * 2, 1)
            
            def forward(self, batch, task='qa'):
                embeddings = self.encoder(batch['entity_ids'], batch['edge_index'])
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
    'lightweight': {
        'dense_model': 'distilbert-base-uncased',
        'hidden_dim': 512,
        'gnn_type': 'gcn',
        'gnn_layers': 2,
        'num_heads': 8,
        'fusion_type': 'cross_attention',
        'fusion_dim': 256,
        'temperature': 0.1,
        'num_entities': 50000,
        'num_relations': 500
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
