from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


ABLATION_MODES = (
    "full",
    "expression_only",
    "vpm_only",
    "no_vpm_pretraining",
    "fixed_cell_fusion",
    "drug_local_only",
    "drug_global_only",
    "fixed_drug_fusion",
    "no_cascaded_attention",
    "no_cell_drug_branch",
    "concat_mlp",
    "no_global_shortcuts",
)


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
        cell_context = self.cell_to_drug(cell_tokens, drug_tokens)
        drug_context = self.drug_to_cell(drug_tokens, cell_tokens)
        return self.cell_self(cell_context), self.drug_self(drug_context)


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
    def __init__(
        self,
        d_model: int,
        num_genes: int,
        dropout: float = 0.1,
        fusion_mode: str = "adaptive",
    ) -> None:
        super().__init__()
        if fusion_mode not in {"adaptive", "fixed", "local_only", "global_only"}:
            raise ValueError(f"Unsupported drug fusion mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.feature_analyzer = None
        self.local_encoder = None
        self.global_context_encoder = None
        if fusion_mode == "adaptive":
            self.feature_analyzer = nn.Sequential(
                nn.Linear(num_genes, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 2),
            )
        if fusion_mode in {"adaptive", "fixed", "local_only"}:
            self.local_encoder = DrugGatedMLPEncoder(d_model=d_model, dropout=dropout)
        if fusion_mode in {"adaptive", "fixed", "global_only"}:
            self.global_context_encoder = ContextGatedDrugEncoder(
                d_model=d_model,
                num_genes=num_genes,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "local_only":
            assert self.local_encoder is not None
            return self.local_encoder(x)
        if self.fusion_mode == "global_only":
            assert self.global_context_encoder is not None
            return self.global_context_encoder(x)
        assert self.local_encoder is not None and self.global_context_encoder is not None
        local_features = self.local_encoder(x)
        global_features = self.global_context_encoder(x)
        if self.fusion_mode == "fixed":
            return 0.5 * local_features + 0.5 * global_features
        assert self.feature_analyzer is not None
        weights = torch.softmax(self.feature_analyzer(x.squeeze(1)), dim=-1)
        local_weight = weights[:, 0].view(-1, 1, 1)
        global_weight = weights[:, 1].view(-1, 1, 1)
        return local_weight * local_features + global_weight * global_features


class SmilesGRUDrugEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        embedding_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, d_model // 2)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_dim * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.long()
        lengths = token_ids.ne(0).sum(dim=1).clamp(min=1).cpu()
        embedded = self.embedding(token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        encoded = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.dropout(self.norm(self.projection(encoded))).unsqueeze(1)


class CellDrugAttentionBranch(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cell_aggregator = nn.MultiheadAttention(d_model, 1, dropout=dropout, batch_first=True)
        self.drug_aggregator = nn.MultiheadAttention(d_model, 1, dropout=dropout, batch_first=True)
        self.cell_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.drug_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cell_drug_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.drug_cell_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, cell_tokens: torch.Tensor, drug_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = cell_tokens.size(0)
        cell_query = self.cell_query.expand(batch_size, -1, -1)
        drug_query = self.drug_query.expand(batch_size, -1, -1)
        cell_representation, _ = self.cell_aggregator(cell_query, cell_tokens, cell_tokens, need_weights=False)
        drug_representation, _ = self.drug_aggregator(drug_query, drug_tokens, drug_tokens, need_weights=False)
        cell_attended, _ = self.cell_drug_attention(
            cell_representation, drug_representation, drug_representation, need_weights=False
        )
        drug_attended, _ = self.drug_cell_attention(
            drug_representation, cell_representation, cell_representation, need_weights=False
        )
        return cell_attended, drug_attended


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
    def __init__(
        self,
        vpm: VPMTransformerEncoder,
        d_model: int,
        num_heads: int,
        dropout: float,
        ffn_factor: float,
        fusion_mode: str = "gated",
    ) -> None:
        super().__init__()
        if fusion_mode not in {"gated", "fixed", "expression_only", "vpm_only"}:
            raise ValueError(f"Unsupported cell fusion mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.vpm = vpm
        self.expr_projection = None
        self.expr_fusion_projection = None
        self.vpm_fusion_projection = None
        self.gate = None
        if fusion_mode != "vpm_only":
            self.expr_projection = nn.Linear(1, d_model)
            self.expr_fusion_projection = nn.Linear(d_model, d_model)
        if fusion_mode != "expression_only":
            self.vpm_fusion_projection = nn.Linear(d_model, d_model)
        if fusion_mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
        self.fusion_attention = SelfAttentionBlock(d_model, num_heads, dropout, ffn_factor)

    def forward(self, expression: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fusion_mode == "expression_only":
            assert self.expr_projection is not None and self.expr_fusion_projection is not None
            expr_tokens = self.expr_projection(expression.transpose(1, 2))
            expr_proj = self.expr_fusion_projection(expr_tokens)
            vpm_dep = expression.squeeze(1).new_zeros(expression.size(0), expression.size(2))
            return self.fusion_attention(expr_proj), vpm_dep

        vpm_tokens, vpm_dep = self.vpm(expression)
        assert self.vpm_fusion_projection is not None
        vpm_proj = self.vpm_fusion_projection(vpm_tokens)
        if self.fusion_mode == "vpm_only":
            return self.fusion_attention(vpm_proj), vpm_dep

        assert self.expr_projection is not None and self.expr_fusion_projection is not None
        expr_tokens = self.expr_projection(expression.transpose(1, 2))
        expr_proj = self.expr_fusion_projection(expr_tokens)
        if self.fusion_mode == "fixed":
            fused = 0.5 * expr_proj + 0.5 * vpm_proj
        else:
            assert self.gate is not None
            gate = self.gate(torch.cat([expr_proj, vpm_proj], dim=-1))
            fused = gate * expr_proj + (1.0 - gate) * vpm_proj
        return self.fusion_attention(fused), vpm_dep


class GlobalCellShortcut(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GlobalDrugShortcut(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

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
        drug_feature: str = "tcs",
        smiles_vocab_size: int = 0,
        smiles_embedding_dim: int = 128,
        smiles_gru_layers: int = 2,
        ablation: str = "full",
    ) -> None:
        super().__init__()
        if drug_feature not in {"tcs", "smiles"}:
            raise ValueError(f"Unsupported drug_feature: {drug_feature}")
        if drug_feature == "smiles" and smiles_vocab_size <= 2:
            raise ValueError("drug_feature=smiles requires a fitted SMILES vocabulary")
        if ablation not in ABLATION_MODES:
            raise ValueError(f"Unsupported ablation: {ablation}")
        if drug_feature != "tcs" and ablation in {"drug_local_only", "drug_global_only", "fixed_drug_fusion"}:
            raise ValueError(f"ablation={ablation} is defined only for the TCS drug encoder")
        self.drug_feature = drug_feature
        self.ablation = ablation
        global_shortcut_dim = 256
        vpm = VPMTransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=encoder_layers,
            ffn_factor=ffn_factor,
            dropout=dropout,
            max_genes=max_genes,
        )
        cell_fusion_mode = {
            "expression_only": "expression_only",
            "vpm_only": "vpm_only",
            "fixed_cell_fusion": "fixed",
        }.get(ablation, "gated")
        self.cell_encoder = CellEncoder(
            vpm=vpm,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            ffn_factor=ffn_factor,
            fusion_mode=cell_fusion_mode,
        )
        if drug_feature == "tcs":
            drug_fusion_mode = {
                "drug_local_only": "local_only",
                "drug_global_only": "global_only",
                "fixed_drug_fusion": "fixed",
            }.get(ablation, "adaptive")
            self.drug_encoder = SafeAdaptiveDrugEncoder(
                d_model=d_model,
                num_genes=tcs_dim,
                dropout=dropout,
                fusion_mode=drug_fusion_mode,
            )
            global_drug_input_dim = tcs_dim
        else:
            self.drug_encoder = SmilesGRUDrugEncoder(
                vocab_size=smiles_vocab_size,
                d_model=d_model,
                embedding_dim=smiles_embedding_dim,
                num_layers=smiles_gru_layers,
                dropout=dropout,
            )
            global_drug_input_dim = d_model
        use_cascade = ablation not in {"no_cascaded_attention", "concat_mlp"}
        use_cell_drug_branch = ablation not in {"no_cell_drug_branch", "concat_mlp"}
        use_global_shortcuts = ablation != "no_global_shortcuts"
        self.cascade_layers = nn.ModuleList(
            [CascadedAttentionLayer(d_model, num_heads, dropout, ffn_factor) for _ in range(num_layers)]
            if use_cascade
            else []
        )
        self.cell_drug_branch = (
            CellDrugAttentionBranch(d_model, num_heads, dropout) if use_cell_drug_branch else None
        )
        self.global_cell_branch = (
            GlobalCellShortcut(expr_dim, global_shortcut_dim, dropout) if use_global_shortcuts else None
        )
        self.global_drug_branch = (
            GlobalDrugShortcut(global_drug_input_dim, global_shortcut_dim, dropout)
            if use_global_shortcuts
            else None
        )
        if ablation == "concat_mlp":
            head_input_dim = d_model * 2 + global_shortcut_dim * 2
        else:
            head_input_dim = d_model * 4
            if use_cell_drug_branch:
                head_input_dim += d_model * 2
            if use_global_shortcuts:
                head_input_dim += global_shortcut_dim * 2
        hidden_dim = max(128, d_model * 2)
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
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

    def forward(self, expression: torch.Tensor, drug_input: torch.Tensor) -> VCATOutput:
        cell_tokens, vpm_dep_pred = self.cell_encoder(expression)
        drug_tokens = self.drug_encoder(drug_input)
        global_drug_input = drug_input.squeeze(1) if self.drug_feature == "tcs" else self._pool(drug_tokens)
        cell_drug_features = []
        if self.cell_drug_branch is not None:
            cell_level, drug_level = self.cell_drug_branch(cell_tokens, drug_tokens)
            cell_drug_features = [cell_level.squeeze(1), drug_level.squeeze(1)]
        for layer in self.cascade_layers:
            cell_tokens, drug_tokens = layer(cell_tokens, drug_tokens)

        pooled_cell = self._pool(cell_tokens)
        pooled_drug = self._pool(drug_tokens)
        global_features = []
        if self.global_cell_branch is not None and self.global_drug_branch is not None:
            global_features = [
                self.global_cell_branch(expression.squeeze(1)),
                self.global_drug_branch(global_drug_input),
            ]
        if self.ablation == "concat_mlp":
            feature_parts = [pooled_cell, pooled_drug, *global_features]
        else:
            feature_parts = [
                pooled_cell,
                pooled_drug,
                pooled_cell * pooled_drug,
                torch.abs(pooled_cell - pooled_drug),
                *cell_drug_features,
                *global_features,
            ]
        features = torch.cat(feature_parts, dim=1)
        logits = self.head(features).squeeze(-1)
        return VCATOutput(logits=logits, vpm_dep_pred=vpm_dep_pred)
