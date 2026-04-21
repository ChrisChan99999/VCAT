from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class VCATOutput:
    logits: torch.Tensor
    vpm_dep_pred: torch.Tensor


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, ffn_factor: float) -> None:
        super().__init__()
        hidden_dim = int(d_model * ffn_factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        normalized = self.norm1(tokens)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        tokens = residual + self.dropout1(attended)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_tokens: torch.Tensor, key_value_tokens: torch.Tensor) -> torch.Tensor:
        normalized_q = self.norm_q(query_tokens)
        normalized_kv = self.norm_kv(key_value_tokens)
        attended, _ = self.attn(normalized_q, normalized_kv, normalized_kv, need_weights=False)
        return query_tokens + self.dropout(attended)


class CascadedAttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, ffn_factor: float) -> None:
        super().__init__()
        self.cell_to_drug = CrossAttentionBlock(d_model, num_heads, dropout)
        self.drug_to_cell = CrossAttentionBlock(d_model, num_heads, dropout)
        self.cell_self = SelfAttentionBlock(d_model, num_heads, dropout, ffn_factor)
        self.drug_self = SelfAttentionBlock(d_model, num_heads, dropout, ffn_factor)

    def forward(self, cell_tokens: torch.Tensor, drug_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cell_tokens = self.cell_to_drug(cell_tokens, drug_tokens)
        drug_tokens = self.drug_to_cell(drug_tokens, cell_tokens)
        return self.cell_self(cell_tokens), self.drug_self(drug_tokens)


class DrugGatedMLPEncoder(nn.Module):
    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = hidden_dim or (d_model // 2)
        self.linear_a1 = nn.Linear(1, hidden_dim)
        self.linear_b1 = nn.Linear(1, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear_a2 = nn.Linear(hidden_dim, d_model)
        self.linear_b2 = nn.Linear(hidden_dim, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        hidden = self.linear_a1(x) * torch.sigmoid(self.linear_b1(x))
        hidden = self.dropout1(self.norm1(hidden))
        out = self.linear_a2(hidden) * torch.sigmoid(self.linear_b2(hidden))
        return self.dropout2(self.norm2(out))


class ContextGatedDrugEncoder(nn.Module):
    def __init__(self, d_model: int, num_genes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.local_encoder = DrugGatedMLPEncoder(d_model=d_model, dropout=dropout)
        self.context_encoder = nn.Sequential(
            nn.Linear(num_genes, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_features = self.local_encoder(x)
        global_context = self.context_encoder(x.squeeze(1)).unsqueeze(1)
        gated = local_features * global_context
        return self.fusion(gated) + local_features


class SafeAdaptiveDrugEncoder(nn.Module):
    def __init__(self, d_model: int, num_genes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_analyzer = nn.Sequential(
            nn.Linear(num_genes, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        self.local_encoder = DrugGatedMLPEncoder(d_model=d_model, dropout=dropout)
        self.global_context_encoder = ContextGatedDrugEncoder(d_model=d_model, num_genes=num_genes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.feature_analyzer(x.squeeze(1)), dim=-1)
        local_weight = weights[:, 0].view(-1, 1, 1)
        global_weight = weights[:, 1].view(-1, 1, 1)
        local_features = self.local_encoder(x)
        global_features = self.global_context_encoder(x)
        return local_weight * local_features + global_weight * global_features


class VPMTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_factor: float,
        dropout: float,
        max_genes: int,
    ) -> None:
        super().__init__()
        self.max_genes = max_genes
        self.input_projection = nn.Linear(1, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, max_genes, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * ffn_factor),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.shared_head = nn.Linear(d_model, 1)

    def forward(self, expression: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = expression.transpose(1, 2)
        _, gene_count, _ = tokens.shape
        if gene_count > self.max_genes:
            raise ValueError(f"gene_count={gene_count} exceeds max_genes={self.max_genes}")
        hidden = self.input_projection(tokens) + self.position_embedding[:, :gene_count, :]
        hidden = self.norm(self.transformer(hidden))
        dependency = self.shared_head(hidden).squeeze(-1)
        return hidden, dependency


class CellEncoder(nn.Module):
    def __init__(self, vpm: VPMTransformerEncoder, d_model: int, num_heads: int, dropout: float, ffn_factor: float) -> None:
        super().__init__()
        self.vpm = vpm
        self.expr_projection = nn.Linear(1, d_model)
        self.expr_fusion_projection = nn.Linear(d_model, d_model)
        self.vpm_fusion_projection = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.fusion_attention = SelfAttentionBlock(d_model, num_heads, dropout, ffn_factor)

    def forward(self, expression: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vpm_tokens, vpm_dep = self.vpm(expression)
        expr_tokens = self.expr_projection(expression.transpose(1, 2))
        expr_proj = self.expr_fusion_projection(expr_tokens)
        vpm_proj = self.vpm_fusion_projection(vpm_tokens)
        gate = self.gate(torch.cat([expr_proj, vpm_proj], dim=-1))
        fused = gate * expr_proj + (1.0 - gate) * vpm_proj
        return self.fusion_attention(fused), vpm_dep


class GlobalShortcut(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float, use_batch_norm: bool) -> None:
        super().__init__()
        layers = [nn.Linear(input_dim, 512)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(512))
        layers.extend([nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, output_dim)])
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class VCATModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        encoder_layers: int,
        ffn_factor: float,
        dropout: float,
        max_genes: int,
        expr_dim: int,
        tcs_dim: int,
    ) -> None:
        super().__init__()
        vpm = VPMTransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=encoder_layers,
            ffn_factor=ffn_factor,
            dropout=dropout,
            max_genes=max_genes,
        )
        self.cell_encoder = CellEncoder(vpm=vpm, d_model=d_model, num_heads=num_heads, dropout=dropout, ffn_factor=ffn_factor)
        self.drug_encoder = SafeAdaptiveDrugEncoder(d_model=d_model, num_genes=tcs_dim, dropout=dropout)
        self.cascade_layers = nn.ModuleList(
            [CascadedAttentionLayer(d_model, num_heads, dropout, ffn_factor) for _ in range(num_layers)]
        )
        self.global_cell_branch = GlobalShortcut(expr_dim, d_model, dropout, use_batch_norm=True)
        self.global_drug_branch = GlobalShortcut(tcs_dim, d_model, dropout, use_batch_norm=False)
        hidden_dim = max(128, d_model * 2)
        self.head = nn.Sequential(
            nn.Linear(d_model * 6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _pool(tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)

    def forward(self, expression: torch.Tensor, tcs: torch.Tensor) -> VCATOutput:
        cell_tokens, vpm_dep_pred = self.cell_encoder(expression)
        drug_tokens = self.drug_encoder(tcs)
        for layer in self.cascade_layers:
            cell_tokens, drug_tokens = layer(cell_tokens, drug_tokens)

        pooled_cell = self._pool(cell_tokens)
        pooled_drug = self._pool(drug_tokens)
        global_cell = self.global_cell_branch(expression.squeeze(1))
        global_drug = self.global_drug_branch(tcs.squeeze(1))
        features = torch.cat(
            [
                pooled_cell,
                pooled_drug,
                pooled_cell * pooled_drug,
                torch.abs(pooled_cell - pooled_drug),
                global_cell,
                global_drug,
            ],
            dim=1,
        )
        logits = self.head(features).squeeze(-1)
        return VCATOutput(logits=logits, vpm_dep_pred=vpm_dep_pred)
