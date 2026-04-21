import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.config import LABEL_LIST, PROMPT_TEMPLATE
from src.llm_scorer import LLMScorer
from src.models.fusion import ConfidenceAwareFusion, RetrievalFeatureEncoder
from src.retrieval.retrieval import KnowledgeAugmentedRetriever


@dataclass
class FusionTrainingConfig:
    model_name: str = "models/lora_llm"
    device: str = "auto"
    batch_size: int = 4
    llm_batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 1e-4
    top_k: int = 10
    alpha: float = 0.7
    lambda_decay: float = 0.1
    gamma: float = 0.5
    initial_beta: float = (
        0.8  # Trust trained LLM more initially; retrieval MLP starts random
    )
    lambda_reg: float = 0.01  # Only used in fusion layer, not doubled
    max_length: int = 2048
    evidence_mode: str = (
        "retrieved"  # "gold" or "retrieved" - per paper, should be "retrieved"
    )
    label_list: List[str] = field(default_factory=lambda: LABEL_LIST)
    retriever_model: str = "bge-vi-base"
    use_class_weights: bool = (
        True  # Address class imbalance with inverse-frequency weighting
    )


def _build_retrieval_features(
    retriever: KnowledgeAugmentedRetriever, text: str, top_k: int, rrf_top_k: int = 20
) -> tuple:
    """Returns (features, retrieved_evidence_text)."""
    # RRF hybrid: top 20 candidates, then Temporal scoring to get top_k
    results = retriever.retrieve(text, top_k=top_k, rrf_top_k=rrf_top_k)
    features = []
    evidence_texts = []

    for r in results:
        features.append([r.score, r.rrf_score, r.recency_score, r.cyclicity_score])
        evidence_texts.append(r.text)

    if len(features) < top_k:
        features.extend([[0.0, 0.0, 0.0, 0.0]] * (top_k - len(features)))

    # Return list of evidence texts (for smart truncation in LLMScorer)
    return np.array(features, dtype=np.float32), evidence_texts


def _normalize_label_to_id(label_value) -> int:
    if isinstance(label_value, (int, float)):
        idx = int(label_value)
        if idx in (0, 1, 2):
            return idx
        raise ValueError(f"Unknown integer label: {label_value}. Expected 0, 1, or 2.")

    label_upper = str(label_value).upper().strip()
    if label_upper in ("TRUE", "ĐÚNG", "DUNG", "SUPPORTED", "LEGIT", "0", "A"):
        return 0
    if label_upper in ("FALSE", "SAI", "REFUTED", "SCAM", "FAKE", "1", "B"):
        return 1
    if label_upper in (
        "NEI",
        "THIEU",
        "NOT ENOUGH INFO",
        "NOT ENOUGH INFORMATION",
        "INSUFFICIENT",
        "2",
        "C",
        "THIẾU THÔNG TIN",
        "CHƯA CHẮC CHẮN",
    ):
        return 2

    if "THIẾU" in label_upper or "THIEU" in label_upper:
        return 2

    raise ValueError(
        f"Unknown label string '{label_value}'. "
        "Expected one of: true, false, nei (or integer 0/1/2)."
    )


def train_fusion_from_dataframe(
    knowledge_base: List[Dict],
    labeled_df,
    config: Optional[FusionTrainingConfig] = None,
    save_path: str = "models/fusion_model.pt",
    resume_checkpoint_path: Optional[str] = None,
    resume_strict: bool = False,
) -> str:
    """
    Train fusion MLP + beta from a labeled pandas DataFrame.

    Per paper Eq.2: pfinal = β·pLM + (1-β)·MLP(pret)
    Uses LOGITS, not probabilities, for proper fusion.

    Args:
        knowledge_base: List of dicts with text/timestamp for retrieval
        labeled_df: DataFrame with text, evidence, label columns
        config: Training configuration
        save_path: Where to save trained model
        resume_checkpoint_path: Optional path to an existing fusion checkpoint (.pt)
        resume_strict: Whether to strictly enforce checkpoint key matching

    evidence_mode options:
        - "retrieved": Use evidence from retriever (paper-accurate)
        - "gold": Use evidence column from dataframe (for debugging)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for fusion training.")

    config = config or FusionTrainingConfig()

    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Training fusion on device: {config.device}")
    logger.info(f"Evidence mode: {config.evidence_mode}")
    logger.info(f"Retriever model: {config.retriever_model}")

    if labeled_df is None or labeled_df.empty:
        raise ValueError("Labeled DataFrame is empty.")

    # Optionally resume from an existing fusion checkpoint.
    resume_payload: Optional[Dict[str, Any]] = None
    if resume_checkpoint_path:
        if not os.path.isfile(resume_checkpoint_path):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_checkpoint_path}"
            )
        logger.info(f"Resuming fusion training from: {resume_checkpoint_path}")
        resume_payload = torch.load(resume_checkpoint_path, map_location="cpu")
        if not isinstance(resume_payload, dict):
            raise ValueError(
                "Resume checkpoint format is invalid. Expected a dictionary payload."
            )

        resume_config = resume_payload.get("config", {})
        if isinstance(resume_config, dict):
            # Keep architecture-compatible values to avoid shape mismatch on load.
            checkpoint_top_k = resume_config.get("top_k")
            if checkpoint_top_k is not None and int(checkpoint_top_k) != int(config.top_k):
                logger.warning(
                    f"Overriding top_k from {config.top_k} -> {checkpoint_top_k} "
                    "to match resume checkpoint architecture."
                )
                config.top_k = int(checkpoint_top_k)

            checkpoint_labels = resume_config.get("label_list")
            if checkpoint_labels is not None:
                checkpoint_labels = [str(x) for x in checkpoint_labels]
                if checkpoint_labels != [str(x) for x in config.label_list]:
                    raise ValueError(
                        "Label list mismatch between current config and resume checkpoint. "
                        f"Current={config.label_list}, checkpoint={checkpoint_labels}"
                    )

            checkpoint_num_classes = resume_config.get("num_classes")
            if checkpoint_num_classes is not None and int(checkpoint_num_classes) != len(
                config.label_list
            ):
                raise ValueError(
                    "num_classes mismatch between config and resume checkpoint. "
                    f"Current={len(config.label_list)}, checkpoint={checkpoint_num_classes}"
                )

            checkpoint_lambda_reg = resume_config.get("lambda_reg")
            if checkpoint_lambda_reg is not None and float(
                checkpoint_lambda_reg
            ) != float(config.lambda_reg):
                logger.warning(
                    f"Overriding lambda_reg from {config.lambda_reg} -> {checkpoint_lambda_reg} "
                    "to continue with checkpoint regularization."
                )
                config.lambda_reg = float(checkpoint_lambda_reg)

    # Initialize retriever with RRF hybrid
    retriever = KnowledgeAugmentedRetriever(
        embedding_model=config.retriever_model,
        alpha=config.alpha,
        lambda_decay=config.lambda_decay,
        gamma=config.gamma,
        use_query_expansion=True,
        rrf_k=60,  # RRF constant
    )
    retriever.index_documents(
        knowledge_base, text_field="text", timestamp_field="timestamp"
    )
    logger.info(f"Indexed {len(knowledge_base)} documents in retriever")

    # Initialize LLM scorer (returns LOGITS per paper Eq.2)
    llm = LLMScorer(
        model_name=config.model_name,
        device=config.device,
        max_length=config.max_length,
        labels=config.label_list,
        prompt_template=PROMPT_TEMPLATE,
    )

    # Initialize retrieval encoder
    retrieval_encoder = RetrievalFeatureEncoder(
        num_retrieved=config.top_k, score_features=4, hidden_dim=64, output_dim=64
    ).to(config.device)

    # Initialize fusion layer
    num_classes = len(config.label_list)
    fusion = ConfidenceAwareFusion(
        retrieval_input_dim=64,
        hidden_dim=128,
        num_classes=num_classes,
        initial_beta=config.initial_beta,
        lambda_reg=config.lambda_reg,  # Regularization only here, not doubled
    ).to(config.device)

    if resume_payload is not None:
        retrieval_state = resume_payload.get("retrieval_encoder")
        fusion_state = resume_payload.get("fusion")
        if retrieval_state is None or fusion_state is None:
            raise ValueError(
                "Resume checkpoint missing required keys: "
                "'retrieval_encoder' and/or 'fusion'."
            )

        retrieval_load = retrieval_encoder.load_state_dict(
            retrieval_state, strict=resume_strict
        )
        fusion_load = fusion.load_state_dict(fusion_state, strict=resume_strict)

        if not resume_strict:
            if retrieval_load.missing_keys or retrieval_load.unexpected_keys:
                logger.warning(
                    "retrieval_encoder checkpoint mismatch "
                    f"(missing={retrieval_load.missing_keys}, "
                    f"unexpected={retrieval_load.unexpected_keys})"
                )
            if fusion_load.missing_keys or fusion_load.unexpected_keys:
                logger.warning(
                    "fusion checkpoint mismatch "
                    f"(missing={fusion_load.missing_keys}, "
                    f"unexpected={fusion_load.unexpected_keys})"
                )

        logger.info("Loaded resume checkpoint weights into fusion components.")

    # Optimizer for fusion components only (LLM is frozen)
    params = list(retrieval_encoder.parameters()) + list(fusion.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    # Prepare data
    texts = labeled_df["text"].tolist()
    gold_evidences = labeled_df["evidence"].tolist()
    # Normalize labels: raw string (true/false/nei) or integer (0/1/2) → int ID
    # ID convention: 0=Đúng (true), 1=Sai (false), 2=NEI (not-enough-info)
    labels = [_normalize_label_to_id(v) for v in labeled_df["label"].tolist()]

    logger.info(f"Training samples: {len(texts)}")

    # Compute class weights to handle imbalance (e.g. False=62%, True=38%)
    label_array = np.array(labels)
    label_counts = np.bincount(label_array, minlength=num_classes)
    logger.info(
        f"Label distribution: { {config.label_list[i]: int(label_counts[i]) for i in range(num_classes)} }"
    )
    if config.use_class_weights:
        # Avoid inf weights when a class is absent in the current training split.
        missing_classes = np.where(label_counts == 0)[0].tolist()
        safe_counts = np.where(label_counts > 0, label_counts, 1).astype(np.float32)
        weights = len(labels) / (num_classes * safe_counts)
        weights = np.where(label_counts > 0, weights, 0.0)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(config.device)
        if missing_classes:
            missing_names = [config.label_list[i] for i in missing_classes]
            logger.warning(
                f"Missing classes in training set: {missing_names}. "
                "Set their class weight to 0.0 to avoid divide-by-zero."
            )
        logger.info(
            f"Class weights (inverse-freq): { {config.label_list[i]: round(float(class_weights[i]), 3) for i in range(num_classes)} }"
        )
    else:
        class_weights = None

    # --- PRE-COMPUTATION PHASE ---
    logger.info("Starting pre-computation of retrieval features and LLM logits...")
    all_retrieval_features = []
    all_llm_logits = []

    # Ensure LLM is in eval mode and cache is disabled
    llm.model.eval()
    llm.model.config.use_cache = False
    for p in llm.model.parameters():
        p.requires_grad_(False)

    # Use CUDA AMP only on CUDA device.
    use_cuda_amp = str(config.device).startswith("cuda") and torch.cuda.is_available()
    amp_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ) if use_cuda_amp else None

    # Process in batches for pre-computation
    for i in range(0, len(texts), config.batch_size):
        batch_texts = texts[i : i + config.batch_size]
        batch_gold_evidences = gold_evidences[i : i + config.batch_size]

        # 1. Retrieval
        batch_feats = []
        batch_retrieved_evidences = []
        for t in batch_texts:
            feats, retrieved_evidence = _build_retrieval_features(
                retriever, t, config.top_k
            )
            batch_feats.append(feats)
            batch_retrieved_evidences.append(
                retrieved_evidence[:3]
            )  # Cắt xuống đúng 3 top evidence để huấn luyện không bị nhiễu LM

        all_retrieval_features.extend(batch_feats)

        # 2. LLM Scoring
        if config.evidence_mode == "retrieved":
            batch_evidences = batch_retrieved_evidences
        else:
            batch_evidences = batch_gold_evidences

        # Micro-batching for LLM inference
        batch_logits_list = []
        with torch.inference_mode():
            amp_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_cuda_amp
                else nullcontext()
            )
            with amp_context:
                for j in range(0, len(batch_texts), config.llm_batch_size):
                    sub_texts = batch_texts[j : j + config.llm_batch_size]
                    sub_evidences = batch_evidences[j : j + config.llm_batch_size]

                    logits = llm.score_logits(sub_texts, sub_evidences)
                    batch_logits_list.append(logits)

        # Concatenate micro-batches
        lm_logits = torch.cat(batch_logits_list, dim=0)
        all_llm_logits.append(lm_logits.cpu())  # Store on CPU to save GPU memory

        if (i // config.batch_size) % 10 == 0:
            logger.info(f"Pre-computed {i + len(batch_texts)}/{len(texts)} samples")

    # Convert to tensors
    tensor_retrieval = torch.tensor(
        np.array(all_retrieval_features), dtype=torch.float32
    )
    tensor_llm_logits = torch.cat(all_llm_logits, dim=0)
    tensor_labels = torch.tensor(labels, dtype=torch.long)

    logger.info("Pre-computation complete. Unloading LLM...")

    # Unload LLM to free memory
    del llm
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    fusion.train()
    retrieval_encoder.train()

    dataset_size = len(texts)
    indices = torch.randperm(dataset_size)

    for epoch in range(config.epochs):
        # Shuffle indices at the start of each epoch
        indices = torch.randperm(dataset_size)

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for i in range(0, dataset_size, config.batch_size):
            batch_indices = indices[i : i + config.batch_size]

            # Move batch to device
            b_retrieval = tensor_retrieval[batch_indices].to(config.device)
            b_llm_logits = tensor_llm_logits[batch_indices].to(config.device)
            b_labels = tensor_labels[batch_indices].to(config.device)

            # Encode retrieval features
            retrieval_features = retrieval_encoder(b_retrieval)

            # Fusion: β·pLM + (1-β)·MLP(pret) per Eq.2
            output = fusion(b_llm_logits, retrieval_features)

            # Loss computation: F.cross_entropy on raw logits [B, num_classes]
            # fused_logits is now [B, 2] for binary and [B, C] for multi-class
            ce_loss = F.cross_entropy(
                output.fused_logits, b_labels, weight=class_weights
            )

            # Add beta regularization (L = CE + λ||β||²)
            beta_reg = fusion.lambda_reg * (fusion.beta**2)
            loss = ce_loss + beta_reg

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()

            # Get predictions using argmax (works for both binary and multi-class)
            preds = torch.argmax(output.final_probs, dim=-1)

            correct += (preds == b_labels).sum().item()
            total += len(b_labels)
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        accuracy = correct / total if total > 0 else 0
        beta_val = fusion.beta.item()

        # Per-class accuracy for debugging (helps detect majority-class collapse)
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for i in range(0, dataset_size, config.batch_size):
                batch_indices = indices[i : i + config.batch_size]
                b_retrieval = tensor_retrieval[batch_indices].to(config.device)
                b_llm_logits = tensor_llm_logits[batch_indices].to(config.device)
                b_labels_eval = tensor_labels[batch_indices]
                retrieval_features_eval = retrieval_encoder(b_retrieval)
                output_eval = fusion(b_llm_logits, retrieval_features_eval)
                preds_eval = torch.argmax(output_eval.final_probs, dim=-1).cpu()
                all_preds.append(preds_eval)
                all_targets.append(b_labels_eval)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        per_class_acc = [
            ((all_preds == i) & (all_targets == i)).sum().item()
            / max(1, (all_targets == i).sum().item())
            for i in range(num_classes)
        ]
        per_class_str = ", ".join(
            f"{config.label_list[i]}: {per_class_acc[i]:.3f}"
            for i in range(num_classes)
        )

        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch + 1}/{config.epochs} - loss: {avg_loss:.4f} - acc: {accuracy:.4f} - β: {beta_val:.4f} - lr: {current_lr:.2e} - per_class: [{per_class_str}]"
        )
        scheduler.step()

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "retrieval_encoder": retrieval_encoder.state_dict(),
            "fusion": fusion.state_dict(),
            "beta": fusion.beta.item(),
            "config": {
                "model_name": config.model_name,
                "retriever_model": config.retriever_model,
                "top_k": config.top_k,
                "num_classes": num_classes,
                "initial_beta": config.initial_beta,
                "lambda_reg": config.lambda_reg,
                "evidence_mode": config.evidence_mode,
                "label_list": config.label_list,
            },
        },
        save_path,
    )

    logger.info(f"Fusion model saved to {save_path}")
    return save_path
