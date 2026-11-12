"""ProFuse-ST: dual graphs, global context, and prototype-enhanced gene fusion."""

import math

import torch
from torch import nn
import torch.nn.functional as F


class GATBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.source_attention = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.target_attention = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.source_attention)
        nn.init.xavier_uniform_(self.target_attention)

    def forward(self, features, neighbors):
        num_spots = features.shape[0]
        values = self.projection(features).view(
            num_spots, self.num_heads, self.head_dim
        ).transpose(0, 1)
        source_score = (values * self.source_attention[:, None]).sum(dim=-1)
        target_score = (values * self.target_attention[:, None]).sum(dim=-1)
        logits = F.leaky_relu(
            source_score.unsqueeze(-1) + target_score[:, neighbors],
            negative_slope=0.2,
        )
        attention = F.softmax(logits, dim=-1)
        context = (
            self.dropout(attention).unsqueeze(-1) * values[:, neighbors]
        ).sum(dim=2)
        context = context.transpose(0, 1).reshape(num_spots, -1)
        return self.output(context), attention.mean(dim=0)


class DualGraphEncoder(nn.Module):
    def __init__(
        self,
        spot_dim,
        hidden_dim,
        num_heads,
        num_neighbors,
        dropout,
        depth=2,
    ):
        super().__init__()
        self.input = nn.Linear(spot_dim, hidden_dim)
        self.spatial_blocks = nn.ModuleList(
            [GATBlock(hidden_dim, num_heads, dropout) for _ in range(depth)]
        )
        self.semantic_blocks = nn.ModuleList(
            [GATBlock(hidden_dim, num_heads, dropout) for _ in range(depth)]
        )
        self.fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _run_branch(self, features, blocks, neighbors):
        attention = None
        for block in blocks:
            update, attention = block(features, neighbors)
            features = features + self.dropout(update)
        return features, attention

    def forward(self, spot_features, graph):
        base_features = self.input(spot_features)
        spatial_neighbors = graph["spatial_neighbors"].to(spot_features.device)
        semantic_neighbors = graph["semantic_neighbors"].to(spot_features.device)
        spatial_features, spatial_attention = self._run_branch(
            base_features, self.spatial_blocks, spatial_neighbors
        )
        semantic_features, semantic_attention = self._run_branch(
            base_features, self.semantic_blocks, semantic_neighbors
        )
        neighborhood_features = self.fusion(
            torch.cat([spatial_features, semantic_features], dim=-1)
        )
        encoded_features = self.norm(
            base_features + self.dropout(neighborhood_features)
        )
        return encoded_features, {
            "base_spot_features": base_features,
            "spatial_spot_features": spatial_features,
            "semantic_spot_features": semantic_features,
            "spatial_neighbors": spatial_neighbors,
            "semantic_neighbors": semantic_neighbors,
            "spatial_attention": spatial_attention,
            "semantic_attention": semantic_attention,
        }


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        attended, _ = self.attention(
            features.unsqueeze(0),
            features.unsqueeze(0),
            features.unsqueeze(0),
            need_weights=False,
        )
        features = self.norm1(features + self.dropout(attended.squeeze(0)))
        return self.norm2(features + self.dropout(self.ffn(features)))


class SinusoidalCoordinateEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout, pe_dim=256):
        super().__init__()
        self.axis_dim = pe_dim // 2
        frequencies = torch.exp(
            torch.arange(0, self.axis_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.axis_dim)
        )
        self.register_buffer("frequencies", frequencies)
        self.mlp = nn.Sequential(
            nn.Linear(pe_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _encode_axis(self, values):
        angles = values.unsqueeze(-1) * self.frequencies.to(values)
        encoding = values.new_zeros(values.shape[0], self.axis_dim)
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles)
        return encoding

    def forward(self, coordinates):
        coordinates = coordinates.float()
        mean = coordinates.mean(dim=0, keepdim=True)
        std = coordinates.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized = (coordinates - mean) / std
        pe_2d = torch.cat(
            [self._encode_axis(normalized[:, 0]), self._encode_axis(normalized[:, 1])],
            dim=-1,
        )
        return self.dropout(self.norm(self.mlp(pe_2d)))


class ProFuseST(nn.Module):
    """The complete ProFuse-ST architecture, without ablation switches."""

    def __init__(self, spot_dim, gene_dim, num_genes, hidden_dim=256,
                 num_programs=32, num_heads=8, dropout=0.1, global_dim=768,
                 local_depth=6, global_depth=6):
        super().__init__()
        if hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if min(spot_dim, gene_dim, num_genes, global_dim, num_programs, local_depth, global_depth) < 1:
            raise ValueError("Feature dimensions, depths, and prototype count must be positive")
        self.spot_encoder = DualGraphEncoder(spot_dim, hidden_dim, num_heads, 6, dropout, local_depth)
        self.global_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout) for _ in range(global_depth)
        ])
        self.position_encoder = SinusoidalCoordinateEncoding(hidden_dim, dropout)
        self.global_context_projection = nn.Linear(global_dim, hidden_dim)
        self.global_context_to_film = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.zeros_(self.global_context_to_film.weight)
        nn.init.zeros_(self.global_context_to_film.bias)
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.local_global_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.spot_fusion_norm = nn.LayerNorm(hidden_dim)
        self.gene_projection = nn.Linear(gene_dim, hidden_dim)
        self.prototypes = nn.Parameter(torch.empty(num_programs, hidden_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.prototype_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.gene_mlp = self._mlp(hidden_dim, dropout)
        self.gene_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_mlp = self._mlp(hidden_dim, dropout)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_genes),
        )

    @staticmethod
    def _mlp(hidden_dim, dropout):
        return nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(),
                             nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim))

    def forward(self, spot_features, gene_embeddings, spot_coords, global_features, graph):
        local, graph_outputs = self.spot_encoder(spot_features, graph)
        base = graph_outputs["base_spot_features"]
        global_spots = base + self.position_encoder(spot_coords)
        for block in self.global_blocks:
            global_spots = block(global_spots)
        if global_features.ndim == 1:
            global_features = global_features.unsqueeze(0)
        if global_features.shape[0] != 1:
            raise ValueError("Provide one whole-section embedding per slide")
        context = self.global_context_projection(global_features).expand(len(spot_features), -1)
        scale, shift = self.global_context_to_film(context).chunk(2, dim=-1)
        global_spots = self.global_norm((1 + torch.tanh(scale)) * global_spots + shift)
        gate = torch.sigmoid(self.local_global_gate(torch.cat([local, global_spots], dim=-1)))
        spots = self.spot_fusion_norm(base + gate * local + (1 - gate) * global_spots)
        genes = self.gene_projection(gene_embeddings).unsqueeze(0)
        prototypes = self.prototypes.unsqueeze(0)
        context, _ = self.prototype_attention(genes, prototypes, prototypes, need_weights=False)
        genes = self.gene_norm(genes + self.gene_mlp(context))
        retrieved, _ = self.cross_attention(spots.unsqueeze(0), genes, genes, need_weights=False)
        fused = self.cross_norm(spots + self.cross_mlp(retrieved.squeeze(0)))
        return {"y_hat": F.softplus(self.prediction_head(fused))}
