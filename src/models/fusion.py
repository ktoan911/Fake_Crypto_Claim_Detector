from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using numpy-based fusion only.")

# PyTorch-based models (only available if torch is installed)
if TORCH_AVAILABLE:

    @dataclass
    class FusionInput:
        """Input container for fusion layer"""

        lm_logits: torch.Tensor
        retrieval_scores: torch.Tensor
        retrieval_features: Optional[torch.Tensor] = None

    @dataclass
    class FusionOutput:
        """Output container for fusion layer"""

        final_probs: torch.Tensor
        fused_logits: torch.Tensor
        lm_weight: torch.Tensor
        retrieval_weight: torch.Tensor

    class RetrievalMLP(nn.Module):
        """
        Two-layer MLP that projects retrieval scores to label space.
        As described in paper: "MLP is a two-layer network that projects
        retrieval scores to the label space"
        """

        def __init__(
            self,
            input_dim: int = 64,
            hidden_dim: int = 128,
            output_dim: int = 1,
            dropout: float = 0.1,
        ):
            super().__init__()

            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

            self._init_weights()

        def _init_weights(self):
            """Initialize weights using Xavier initialization"""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layers(x)

    class ConfidenceAwareFusion(nn.Module):
        """
        Implements Equation (2) from the paper:
        pfinal(y|q, D) = σ(β · pLM + (1 − β) · MLP(pret))

        Where σ is Sigmoid for binary classification (num_classes=2).
        For binary: MLP outputs 1 logit; positive/negative probs are derived via softmax.

        With contrastive loss from Equation (3):
        L = -log(e^sp / Σe^sn) + λ||β||²
        """

        def __init__(
            self,
            retrieval_input_dim: int = 64,
            hidden_dim: int = 128,
            num_classes: int = 2,
            initial_beta: float = 0.5,
            lambda_reg: float = 0.01,
            learn_beta: bool = True,
            normalize_branch_logits: bool = False,
            adaptive_beta: bool = False,
            beta_hidden_dim: int = 16,
        ):
            super().__init__()

            self.num_classes = num_classes
            self.lambda_reg = lambda_reg
            self.is_binary = num_classes == 2
            self.normalize_branch_logits = normalize_branch_logits
            self.adaptive_beta = adaptive_beta

            # Trainable gating parameter β
            # We use a logit parameter and apply sigmoid in forward() to ensure β ∈ [0, 1]
            self._beta_logit = nn.Parameter(
                torch.tensor(self._inverse_sigmoid(initial_beta)),
                requires_grad=learn_beta,
            )

            # Trainable per-branch temperature (log-parameterized so it stays > 0).
            # Fixed across all samples — unlike per-sample std standardization, this
            # can't invert the confidence/correctness relationship (dividing a
            # confident sample's own large logit gap by its own std shrinks it,
            # while dividing an uncertain sample's small gap by its own small std
            # inflates it). A single learned scalar calibrates each branch's
            # average scale without that per-sample side effect.
            self._lm_log_temp = nn.Parameter(torch.zeros(()))
            self._retrieval_log_temp = nn.Parameter(torch.zeros(()))

            # MLP for projecting retrieval scores
            # For binary classification, output 1 logit; else output num_classes logits
            mlp_output_dim = 1 if self.is_binary else num_classes
            self.retrieval_mlp = RetrievalMLP(
                input_dim=retrieval_input_dim,
                hidden_dim=hidden_dim,
                output_dim=mlp_output_dim,
            )

            # Optional per-sample gate: β_i = sigmoid(β_global_logit + g(features_i))
            # This lets fusion trust different branches for different claims.
            if self.adaptive_beta:
                gate_input_dim = (self.num_classes * 2) + 2
                self.beta_gate = nn.Sequential(
                    nn.Linear(gate_input_dim, beta_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(beta_hidden_dim, 1),
                )
            else:
                self.beta_gate = None

            activation_type = "sigmoid" if self.is_binary else "softmax"
            logger.info(
                "ConfidenceAwareFusion initialized: "
                f"β={initial_beta}, λ={lambda_reg}, num_classes={num_classes}, "
                f"activation={activation_type}, normalize_branch_logits={normalize_branch_logits}, "
                f"adaptive_beta={adaptive_beta}"
            )

        def _inverse_sigmoid(self, x: float) -> float:
            """Inverse sigmoid for initialization"""
            x = np.clip(x, 1e-6, 1 - 1e-6)
            return np.log(x / (1 - x))

        @property
        def beta(self) -> torch.Tensor:
            """Get the current gating parameter β. Guaranteed to be in [0, 1]."""
            return torch.sigmoid(self._beta_logit)

        @property
        def lm_temperature(self) -> torch.Tensor:
            """Learned LM-branch temperature. Guaranteed > 0 via exp()."""
            return torch.exp(self._lm_log_temp)

        @property
        def retrieval_temperature(self) -> torch.Tensor:
            """Learned retrieval-branch temperature. Guaranteed > 0 via exp()."""
            return torch.exp(self._retrieval_log_temp)

        def _normalize_logits(self, logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
            """Center a branch's logits and rescale by its learned, fixed temperature.

            Unlike per-sample std standardization, dividing by a single value
            learned across the whole training set can't invert the
            confidence/correctness relationship: it can't shrink a genuinely
            confident sample's gap just because that one sample happens to have
            a large spread, nor inflate an uncertain sample's gap just because
            that one sample happens to have a small spread.
            """
            centered = logits - logits.mean(dim=-1, keepdim=True)
            return centered / temperature

        def _normalized_entropy(self, probs: torch.Tensor) -> torch.Tensor:
            eps = 1e-8
            entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1, keepdim=True)
            max_entropy = np.log(max(2, self.num_classes))
            return entropy / max(max_entropy, eps)

        def _compute_beta_for_batch(
            self, lm_logits: torch.Tensor, retrieval_logits_label_space: torch.Tensor
        ) -> torch.Tensor:
            if not self.adaptive_beta or self.beta_gate is None:
                return self.beta

            lm_probs = torch.softmax(lm_logits, dim=-1)
            retrieval_probs = torch.softmax(retrieval_logits_label_space, dim=-1)
            lm_entropy = self._normalized_entropy(lm_probs)
            retrieval_entropy = self._normalized_entropy(retrieval_probs)

            gate_input = torch.cat(
                [lm_probs, retrieval_probs, lm_entropy, retrieval_entropy], dim=-1
            )
            beta_delta = self.beta_gate(gate_input)
            # Broadcast-ready shape [B, 1]
            return torch.sigmoid(self._beta_logit + beta_delta)

        def forward(
            self, lm_logits: torch.Tensor, retrieval_features: torch.Tensor
        ) -> FusionOutput:
            """Forward pass implementing Equation (2): β·pLM + (1-β)·MLP(pret)."""

            batch_size = lm_logits.size(0)
            beta = self.beta

            # Project retrieval features to label space
            retrieval_logits = self.retrieval_mlp(retrieval_features)
            retrieval_logits_label_space = retrieval_logits

            if self.is_binary:
                # Binary classification: treat as 2-class softmax (same as multi-class)
                # lm_logits: [B, 2] in LABEL_LIST order [positive, negative]
                assert lm_logits.size(1) == 2, (
                    f"Binary mode: lm_logits should be [B, 2], got {lm_logits.shape}"
                )

                # For binary, MLP outputs 1 logit → expand to 2 logits [pos, neg]
                # retrieval_logits: [B, 1] → treat as positive logit; negative = -positive
                assert retrieval_logits.size() == (batch_size, 1), (
                    f"Binary mode: retrieval_logits should be [B, 1], got {retrieval_logits.shape}"
                )

                # Expand retrieval to 2 logits in label order: [r, -r]
                retrieval_logits_2 = torch.cat(
                    [retrieval_logits, -retrieval_logits], dim=-1
                )  # [B, 2]
                retrieval_logits_label_space = retrieval_logits_2

                if self.normalize_branch_logits:
                    lm_logits = self._normalize_logits(lm_logits, self.lm_temperature)
                    retrieval_logits_label_space = self._normalize_logits(
                        retrieval_logits_label_space, self.retrieval_temperature
                    )

                beta = self._compute_beta_for_batch(
                    lm_logits, retrieval_logits_label_space
                )
                fused_logits = beta * lm_logits + (1 - beta) * retrieval_logits_label_space

                # Apply softmax to get probabilities
                final_probs = torch.softmax(fused_logits, dim=-1)  # [B, 2]
                # final_probs[:, 0] = P(positive), final_probs[:, 1] = P(negative)

            else:
                # Multi-class: use softmax
                assert lm_logits.size(1) == self.num_classes, (
                    f"lm_logits shape mismatch: expected [B, {self.num_classes}], got {lm_logits.shape}"
                )

                assert retrieval_logits.size() == (batch_size, self.num_classes), (
                    f"retrieval_logits shape mismatch: expected [{batch_size}, {self.num_classes}], got {retrieval_logits.shape}"
                )

                if self.normalize_branch_logits:
                    lm_logits = self._normalize_logits(lm_logits, self.lm_temperature)
                    retrieval_logits_label_space = self._normalize_logits(
                        retrieval_logits_label_space, self.retrieval_temperature
                    )

                beta = self._compute_beta_for_batch(
                    lm_logits, retrieval_logits_label_space
                )
                fused_logits = beta * lm_logits + (1 - beta) * retrieval_logits_label_space
                final_probs = torch.softmax(fused_logits, dim=-1)

            return FusionOutput(
                final_probs=final_probs,
                fused_logits=fused_logits,
                lm_weight=beta.detach().mean(),
                retrieval_weight=(1 - beta).detach().mean(),
            )

        def compute_contrastive_loss(
            self,
            positive_scores: torch.Tensor,
            negative_scores: torch.Tensor,
            temperature: float = 1.0,
        ) -> torch.Tensor:
            """Compute contrastive loss from Equation (3)."""
            sp = positive_scores / temperature
            sn = negative_scores / temperature

            numerator = torch.exp(sp)
            denominator = numerator + torch.sum(torch.exp(sn), dim=-1, keepdim=True)

            contrastive_loss = -torch.log(numerator / (denominator + 1e-8))
            contrastive_loss = contrastive_loss.mean()

            beta_reg = self.lambda_reg * (self.beta**2)

            return contrastive_loss + beta_reg

    class RetrievalFeatureEncoder(nn.Module):
        """Encodes retrieval results into features for fusion.

        Two optional branches:
        - Score branch: attention-weighted scoring features → output_dim
        - Interaction branch (if interaction_dim > 0):
            claim-evidence interaction [q⊙mean_d, |q-mean_d|] → output_dim
            Uses a single linear projection + dropout to avoid overfitting on
            small datasets.  The interaction signal encodes whether the claim
            ALIGNS or CONTRADICTS the retrieved evidence per semantic dimension.
        Both branches are concatenated then projected back to output_dim.
        """

        def __init__(
            self,
            num_retrieved: int = 5,
            score_features: int = 5,
            hidden_dim: int = 64,
            output_dim: int = 64,
            interaction_dim: int = 0,
        ):
            super().__init__()

            self.num_retrieved = num_retrieved
            self.interaction_dim = interaction_dim
            input_dim = num_retrieved * score_features

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

            self.attention = nn.Sequential(
                nn.Linear(score_features, 16), nn.Tanh(), nn.Linear(16, 1)
            )

            if interaction_dim > 0:
                # Single linear layer + dropout (no hidden layer to limit overfitting
                # when training set is small, e.g. ~274 samples).
                self.interaction_proj = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(interaction_dim, output_dim),
                )
                # Fuses score branch + interaction branch → output_dim
                self.fusion_proj = nn.Linear(output_dim * 2, output_dim)
            else:
                self.interaction_proj = None
                self.fusion_proj = None

        def forward(
            self,
            retrieval_scores: torch.Tensor,
            interaction_features: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Args:
                retrieval_scores:     [B, num_retrieved, score_features]
                interaction_features: [B, interaction_dim] or None
            Returns:
                [B, output_dim]
            """
            batch_size = retrieval_scores.size(0)

            attn_logits = self.attention(retrieval_scores)
            attn_weights = F.softmax(attn_logits, dim=1)

            weighted = retrieval_scores * attn_weights
            flat = weighted.view(batch_size, -1)
            score_out = self.encoder(flat)  # [B, output_dim]

            if self.interaction_proj is not None and interaction_features is not None:
                int_out = self.interaction_proj(interaction_features)  # [B, output_dim]
                combined = torch.cat([score_out, int_out], dim=-1)    # [B, output_dim*2]
                return self.fusion_proj(combined)                       # [B, output_dim]

            return score_out
