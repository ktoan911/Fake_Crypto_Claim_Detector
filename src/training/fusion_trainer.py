import gc
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
    beta_lr_multiplier: float = (
        10.0  # Aggressive β updates so it can escape 0.8 plateau when LLM dominates.
    )
    top_k: int = 10
    alpha: float = 0.7
    lambda_decay: float = 0.1
    gamma: float = 0.5
    initial_beta: float = 0.7
    lambda_reg: float = 0.0
    max_length: int = 8192
    evidence_mode: str = (
        "retrieved"  # "gold" or "retrieved". Must match serving (api_server.py uses retrieved
        # evidence), otherwise beta learns to over-trust the LLM branch on clean gold evidence
        # and then fails at serve time when retrieval evidence is noisy/truncated.
    )
    llm_evidence_top_k: int = int(
        os.getenv("FUSION_LLM_EVIDENCE_TOP_K", "10")
    )  # Must match FUSION_LLM_EVIDENCE_TOP_K used at serving time.
    label_list: List[str] = field(default_factory=lambda: LABEL_LIST)
    retriever_model: str = "bge-vi-base"
    use_class_weights: bool = (
        True  # Address class imbalance with inverse-frequency weighting
    )
    align_runtime_with_resume_checkpoint: bool = (
        True  # Keep LLM/retriever/evidence settings consistent with resumed checkpoint.
    )
    normalize_branch_logits: bool = (
        True  # Standardize LLM/retrieval logits before weighted fusion.
    )
    adaptive_beta: bool = (
        True  # Learn per-sample beta offsets from branch confidence patterns.
    )
    retrieval_aux_loss_weight: float = (
        0.4  # Stronger supervision: force retrieval MLP to learn label prediction from interaction features.
    )
    save_best_checkpoint: bool = (
        True  # Save best epoch on training-set metrics instead of last epoch only.
    )
    use_nli: bool = True
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    nli_batch_size: int = 64


def _build_retrieval_features(
    retriever: KnowledgeAugmentedRetriever, text: str, top_k: int, rrf_top_k: int = 20
) -> tuple:
    """
    Nhận vào 1 câu claim, đi tìm bằng chứng, và số hoá
    """
    results = retriever.retrieve(text, top_k=top_k, rrf_top_k=rrf_top_k)
    features = []
    doc_embs = []
    evidence_texts = []


    # Duyệt qua các kết quả, lấy ra điểm số rời rạc (Base, RRF, Recency, Cyclicity, Cosine).
    for r in results:
        features.append([r.score, r.rrf_score, r.recency_score, r.cyclicity_score, r.cosine_similarity])
        evidence_texts.append(r.text)
        if r.embedding is not None:
            doc_embs.append(r.embedding)

    pad = top_k - len(features)
    if pad > 0:
        features.extend([[0.0, 0.0, 0.0, 0.0, 0.0]] * pad)

    # Tính toán sự tương tác giữa claim và evidence
    interaction = None
    q_emb = getattr(retriever, "_last_query_embedding", None)
    if q_emb is not None:
        if doc_embs:
            mean_d = np.array(doc_embs, dtype=np.float32).mean(axis=0)
            interaction = np.concatenate(
                [q_emb * mean_d, np.abs(q_emb - mean_d)], dtype=np.float32
            )
        else:
            # Nếu không tìm thấy doc nào, gán toàn bộ bằng 0.
            interaction = np.zeros(2 * len(q_emb), dtype=np.float32)

    return np.array(features, dtype=np.float32), interaction, evidence_texts


def _save_training_curves(
    history: Dict[str, list],
    save_path: str,
    best_epoch: int = -1,
) -> List[str]:
    """Plot per-epoch training metrics as separate PNG files next to save_path.

    Returns list of image paths written (empty if matplotlib unavailable or
    no metrics).
    """
    if not history or not history.get("loss"):
        logger.warning("No training history to plot.")
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")  # Headless: don't require a display.
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping training curves.")
        return []

    out_dir = os.path.dirname(save_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(history["loss"]) + 1))
    suffix = f" — best epoch: {best_epoch}" if best_epoch > 0 else ""
    written: List[str] = []

    def _mark_best(ax):
        if best_epoch > 0:
            ax.axvline(best_epoch, color="gray", linestyle=":", alpha=0.6, label=f"best ep={best_epoch}")

    # 1) Losses
    img_path = os.path.join(out_dir, "loss_curve.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["loss"], label="Total loss", marker="o", markersize=3)
    ax.plot(
        epochs,
        history["retrieval_aux_loss"],
        label="Retrieval aux loss",
        marker="s",
        markersize=3,
        alpha=0.7,
    )
    _mark_best(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Losses{suffix}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(img_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(img_path)
    logger.info(f"Loss curve saved to {img_path}")

    # 2) Branch accuracies
    img_path = os.path.join(out_dir, "accuracy_curve.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["acc"], label="Fusion acc", marker="o", markersize=3, linewidth=2)
    ax.plot(epochs, history["llm_acc"], label="LLM-only acc", linestyle="--", alpha=0.8)
    ax.plot(
        epochs,
        history["retrieval_acc"],
        label="Retrieval-branch acc",
        linestyle=":",
        alpha=0.8,
    )
    _mark_best(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Branch accuracies{suffix}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(img_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(img_path)
    logger.info(f"Accuracy curve saved to {img_path}")

    # 3) β + learning rate (twin axis)
    img_path = os.path.join(out_dir, "beta_lr_curve.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["beta"], color="tab:purple", marker="o", markersize=3, label="β (LLM weight)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("β", color="tab:purple")
    ax.tick_params(axis="y", labelcolor="tab:purple")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(epochs, history["lr"], color="tab:orange", linestyle="--", label="lr")
    ax2.set_ylabel("Learning rate", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_yscale("log")

    _mark_best(ax)

    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, loc="best", fontsize=9)
    ax.set_title(f"β evolution + LR schedule{suffix}")
    plt.tight_layout()
    plt.savefig(img_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(img_path)
    logger.info(f"β + LR curve saved to {img_path}")

    return written


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
    save_path: str = "models/acf_fusion_model.pt",
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
    logger.info(f"fusion_trainer source: {__file__}")
    logger.info(f"Evidence mode (requested): {config.evidence_mode}")
    logger.info(f"Retriever model (requested): {config.retriever_model}")
    logger.info(
        "Trainer flags: "
        f"normalize_branch_logits={config.normalize_branch_logits}, "
        f"adaptive_beta={config.adaptive_beta}, "
        f"retrieval_aux_loss_weight={config.retrieval_aux_loss_weight}, "
        f"save_best_checkpoint={config.save_best_checkpoint}, "
        f"beta_lr_multiplier={config.beta_lr_multiplier}"
    )

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

            # Override use_nli / nli_model if checkpoint was trained with NLI
            checkpoint_nli_model = resume_config.get("nli_model")
            checkpoint_score_features = resume_config.get("score_features")
            if checkpoint_score_features is not None:
                expected_sf = 8 if config.use_nli else 5
                if int(checkpoint_score_features) != expected_sf:
                    logger.warning(
                        f"Overriding score_features {expected_sf} -> {checkpoint_score_features} "
                        "to match resume checkpoint architecture."
                    )
                    config.use_nli = int(checkpoint_score_features) == 8
            if checkpoint_nli_model and not config.use_nli:
                logger.warning(
                    f"Checkpoint was trained with NLI ({checkpoint_nli_model}). "
                    "Forcing use_nli=True to match architecture."
                )
                config.use_nli = True
                config.nli_model = checkpoint_nli_model

            if config.align_runtime_with_resume_checkpoint:
                runtime_keys = (
                    "model_name",
                    "retriever_model",
                    "normalize_branch_logits",
                    "adaptive_beta",
                    "retrieval_aux_loss_weight",
                )
                for key in runtime_keys:
                    checkpoint_value = resume_config.get(key)
                    if checkpoint_value is None:
                        continue

                    # Skip unusable absolute local paths baked into another runtime/container.
                    if isinstance(checkpoint_value, str):
                        expanded_value = os.path.expanduser(checkpoint_value)
                        if os.path.isabs(expanded_value) and not os.path.exists(
                            expanded_value
                        ):
                            logger.warning(
                                f"Skipping resume override for {key}='{checkpoint_value}' "
                                "because the absolute local path does not exist in this environment."
                            )
                            continue

                    current_value = getattr(config, key, None)
                    if current_value != checkpoint_value:
                        logger.warning(
                            f"Overriding {key} from {current_value} -> {checkpoint_value} "
                            "to match resume checkpoint runtime."
                        )
                        setattr(config, key, checkpoint_value)

    logger.info(
        "Effective runtime after resume-merge: "
        f"evidence_mode={config.evidence_mode}, "
        f"model_name={config.model_name}, "
        f"retriever_model={config.retriever_model}"
    )

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

    # Determine score_features upfront (5 base + 3 NLI if use_nli)
    score_features = 8 if config.use_nli else 5

    # Initialize retrieval encoder
    raw_emb_dim = retriever.embedding_dim if retriever.encoder is not None else 0
    interaction_dim = 2 * raw_emb_dim  # q⊙mean_d + |q-mean_d|
    retrieval_encoder = RetrievalFeatureEncoder(
        num_retrieved=config.top_k,
        score_features=score_features,
        hidden_dim=64,
        output_dim=64,
        interaction_dim=interaction_dim,
    ).to(config.device)
    logger.info(
        f"RetrievalFeatureEncoder: score_features={score_features}, "
        f"raw_emb_dim={raw_emb_dim}, interaction_dim={interaction_dim}"
    )

    # Initialize fusion layer
    num_classes = len(config.label_list)
    fusion = ConfidenceAwareFusion(
        retrieval_input_dim=64,
        hidden_dim=128,
        num_classes=num_classes,
        initial_beta=config.initial_beta,
        lambda_reg=config.lambda_reg,  # Regularization only here, not doubled
        normalize_branch_logits=bool(config.normalize_branch_logits),
        adaptive_beta=bool(config.adaptive_beta),
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
    fusion_non_beta_params = [
        p for name, p in fusion.named_parameters() if name != "_beta_logit"
    ]
    base_params = list(retrieval_encoder.parameters()) + fusion_non_beta_params
    beta_lr = config.learning_rate * max(float(config.beta_lr_multiplier), 0.0)
    optimizer_groups = [
        {"params": base_params, "lr": config.learning_rate, "weight_decay": 0.01}
    ]
    if fusion._beta_logit.requires_grad:
        optimizer_groups.append(
            {
                "params": [fusion._beta_logit],
                "lr": beta_lr if beta_lr > 0 else config.learning_rate,
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(optimizer_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )
    beta_group_lr = (
        optimizer.param_groups[1]["lr"]
        if len(optimizer.param_groups) > 1
        else optimizer.param_groups[0]["lr"]
    )
    logger.info(
        f"Optimizer LRs -> base: {config.learning_rate:.2e}, beta: {beta_group_lr:.2e}"
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

    # ---------------------------------------------------------------
    # PHASE 1: Retrieval features (base 5 per doc)
    # ---------------------------------------------------------------
    logger.info("Pre-computation phase 1/3: retrieval features...")
    all_retrieval_base: List[np.ndarray] = []   # each [top_k, 5]
    all_interactions: List[Optional[np.ndarray]] = []
    all_evidence_raw: List[List[str]] = []       # raw texts for NLI
    all_retrieved_evidences: List[List[str]] = []  # top-3 for LLM

    for idx, text in enumerate(texts):
        feats, interaction, evidence = _build_retrieval_features(retriever, text, config.top_k)
        all_retrieval_base.append(feats)
        all_interactions.append(interaction)
        all_evidence_raw.append(evidence)
        all_retrieved_evidences.append(evidence[: config.llm_evidence_top_k])
        if (idx + 1) % 50 == 0:
            logger.info(f"  Retrieved {idx + 1}/{len(texts)} samples")

    # ---------------------------------------------------------------
    # PHASE 2: NLI features (optional, loaded/unloaded before LLM)
    # ---------------------------------------------------------------
    all_retrieval_features: List[np.ndarray] = all_retrieval_base

    if config.use_nli:
        from src.models.nli_scorer import NLIScorer

        logger.info("Pre-computation phase 2/3: NLI stance features...")
        nli_scorer = NLIScorer(
            model_name=config.nli_model,
            device=config.device,
            batch_size=config.nli_batch_size,
        )
        nli_scorer.load()

        flat_docs: List[str] = []
        flat_claims: List[str] = []
        for text, evidences in zip(texts, all_evidence_raw):
            for doc in evidences:
                flat_docs.append(doc)
                flat_claims.append(text)

        logger.info(f"  Running NLI on {len(flat_docs)} (claim, evidence) pairs...")
        nli_flat = nli_scorer.score(premises=flat_docs, hypotheses=flat_claims)
        nli_scorer.unload()
        del nli_scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cursor = 0
        all_retrieval_features = []
        for base_feats, evidences in zip(all_retrieval_base, all_evidence_raw):
            n_real = len(evidences)
            nli_padded = np.full((config.top_k, 3), 1.0 / 3.0, dtype=np.float32)
            if n_real > 0:
                nli_padded[:n_real] = nli_flat[cursor : cursor + n_real]
            cursor += n_real
            all_retrieval_features.append(
                np.concatenate([base_feats, nli_padded], axis=-1)  # [top_k, 8]
            )
        logger.info(f"  NLI merged. score_features={score_features}, pairs={len(flat_docs)}")

    # ---------------------------------------------------------------
    # PHASE 3: LLM logits (loaded here to avoid co-loading with NLI)
    # ---------------------------------------------------------------
    logger.info("Pre-computation phase 3/3: LLM logits...")
    llm = LLMScorer(
        model_name=config.model_name,
        device=config.device,
        max_length=config.max_length,
        labels=config.label_list,
        prompt_template=PROMPT_TEMPLATE,
    )
    llm.model.eval()
    llm.model.config.use_cache = False
    for p in llm.model.parameters():
        p.requires_grad_(False)

    use_cuda_amp = str(config.device).startswith("cuda") and torch.cuda.is_available()
    amp_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ) if use_cuda_amp else None

    all_llm_logits = []
    for i in range(0, len(texts), config.batch_size):
        batch_texts = texts[i : i + config.batch_size]
        if config.evidence_mode == "retrieved":
            batch_evidences = all_retrieved_evidences[i : i + config.batch_size]
        else:
            batch_evidences = gold_evidences[i : i + config.batch_size]

        batch_logits_list = []
        with torch.inference_mode():
            amp_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_cuda_amp
                else nullcontext()
            )
            with amp_context:
                for j in range(0, len(batch_texts), config.llm_batch_size):
                    logits = llm.score_logits(
                        batch_texts[j : j + config.llm_batch_size],
                        batch_evidences[j : j + config.llm_batch_size],
                    )
                    batch_logits_list.append(logits)

        all_llm_logits.append(torch.cat(batch_logits_list, dim=0).cpu())
        if (i // config.batch_size) % 10 == 0:
            logger.info(f"  LLM scored {i + len(batch_texts)}/{len(texts)} samples")

    # Convert to tensors
    tensor_retrieval = torch.tensor(np.array(all_retrieval_features), dtype=torch.float32)
    tensor_llm_logits = torch.cat(all_llm_logits, dim=0)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    llm_baseline_acc = (
        (torch.argmax(tensor_llm_logits, dim=-1) == tensor_labels).float().mean().item()
    )
    logger.info(f"LLM baseline acc: {llm_baseline_acc:.4f}")

    # Build interaction tensor [N, 2*emb_dim]
    tensor_interactions: Optional[torch.Tensor] = None
    _non_none_interactions = [x for x in all_interactions if x is not None]
    if _non_none_interactions:
        emb_shape = _non_none_interactions[0].shape[0]
        _filled = [
            x if x is not None else np.zeros(emb_shape, dtype=np.float32)
            for x in all_interactions
        ]
        tensor_interactions = torch.tensor(np.array(_filled, dtype=np.float32), dtype=torch.float32)
        logger.info(f"Interaction features tensor: {tensor_interactions.shape}")

    logger.info("Pre-computation complete. Unloading LLM...")
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    fusion.train()
    retrieval_encoder.train()

    dataset_size = len(texts)
    indices = torch.randperm(dataset_size)
    best_epoch = -1
    best_accuracy = float("-inf")
    best_loss = float("inf")
    best_state: Optional[Dict[str, Any]] = None

    # Per-epoch metric history for plotting after training.
    history: Dict[str, list] = {
        "loss": [],
        "retrieval_aux_loss": [],
        "acc": [],
        "llm_acc": [],
        "retrieval_acc": [],
        "beta": [],
        "lr": [],
        "per_class": [],
    }

    for epoch in range(config.epochs):
        # Shuffle indices at the start of each epoch
        indices = torch.randperm(dataset_size)

        total_loss = 0.0
        total_retrieval_aux_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for i in range(0, dataset_size, config.batch_size):
            batch_indices = indices[i : i + config.batch_size]

            # Move batch to device
            b_retrieval = tensor_retrieval[batch_indices].to(config.device)
            b_llm_logits = tensor_llm_logits[batch_indices].to(config.device)
            b_labels = tensor_labels[batch_indices].to(config.device)
            b_interaction = (
                tensor_interactions[batch_indices].to(config.device)
                if tensor_interactions is not None else None
            )

            # Encode retrieval features
            retrieval_features = retrieval_encoder(b_retrieval, b_interaction)

            # Fusion: β·pLM + (1-β)·MLP(pret) per Eq.2
            output = fusion(b_llm_logits, retrieval_features)

            # Loss computation: F.cross_entropy on raw logits [B, num_classes]
            # fused_logits is now [B, 2] for binary and [B, C] for multi-class
            ce_loss = F.cross_entropy(
                output.fused_logits, b_labels, weight=class_weights
            )

            retrieval_logits_aux = fusion.retrieval_mlp(retrieval_features)
            if fusion.is_binary:
                retrieval_logits_aux = torch.cat(
                    [retrieval_logits_aux, -retrieval_logits_aux], dim=-1
                )
            if fusion.normalize_branch_logits:
                retrieval_logits_aux = fusion._normalize_logits(retrieval_logits_aux)
            retrieval_aux_loss = F.cross_entropy(
                retrieval_logits_aux, b_labels, weight=class_weights
            )

            # Add beta regularization (L = CE + λ||β||²)
            beta_reg = fusion.lambda_reg * (fusion.beta**2)
            loss = (
                ce_loss
                + float(config.retrieval_aux_loss_weight) * retrieval_aux_loss
                + beta_reg
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if num_batches < 20:
                beta_grad = fusion._beta_logit.grad
                logger.debug(
                    f"[beta_debug] epoch={epoch} step={num_batches}"
                    f" beta_logit={fusion._beta_logit.item():.4f}"
                    f" beta_logit_grad={float(beta_grad.item()) if beta_grad is not None else None}"
                    f" beta={fusion.beta.item():.4f}"
                )
            torch.nn.utils.clip_grad_norm_(
                list(retrieval_encoder.parameters()) + list(fusion.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_retrieval_aux_loss += retrieval_aux_loss.item()

            # Get predictions using argmax (works for both binary and multi-class)
            preds = torch.argmax(output.final_probs, dim=-1)

            correct += (preds == b_labels).sum().item()
            total += len(b_labels)
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_retrieval_aux_loss = total_retrieval_aux_loss / max(1, num_batches)
        accuracy = correct / total if total > 0 else 0
        beta_val = fusion.beta.item()

        # Per-class and branch-wise accuracy for debugging (helps detect branch collapse)
        fusion.eval()
        retrieval_encoder.eval()
        all_preds = []
        all_targets = []
        all_llm_preds = []
        all_retrieval_preds = []
        with torch.no_grad():
            for i in range(0, dataset_size, config.batch_size):
                batch_indices = indices[i : i + config.batch_size]
                b_retrieval = tensor_retrieval[batch_indices].to(config.device)
                b_llm_logits = tensor_llm_logits[batch_indices].to(config.device)
                b_labels_eval = tensor_labels[batch_indices]
                b_interaction_eval = (
                    tensor_interactions[batch_indices].to(config.device)
                    if tensor_interactions is not None else None
                )
                retrieval_features_eval = retrieval_encoder(b_retrieval, b_interaction_eval)
                output_eval = fusion(b_llm_logits, retrieval_features_eval)
                preds_eval = torch.argmax(output_eval.final_probs, dim=-1).cpu()
                all_preds.append(preds_eval)
                all_targets.append(b_labels_eval)

                llm_preds_eval = torch.argmax(b_llm_logits, dim=-1).cpu()
                all_llm_preds.append(llm_preds_eval)

                retrieval_logits_eval = fusion.retrieval_mlp(retrieval_features_eval)
                if fusion.is_binary:
                    retrieval_logits_eval = torch.cat(
                        [retrieval_logits_eval, -retrieval_logits_eval], dim=-1
                    )
                if fusion.normalize_branch_logits:
                    retrieval_logits_eval = fusion._normalize_logits(retrieval_logits_eval)
                retrieval_preds_eval = torch.argmax(retrieval_logits_eval, dim=-1).cpu()
                all_retrieval_preds.append(retrieval_preds_eval)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_llm_preds = torch.cat(all_llm_preds)
        all_retrieval_preds = torch.cat(all_retrieval_preds)
        per_class_acc = [
            ((all_preds == i) & (all_targets == i)).sum().item()
            / max(1, (all_targets == i).sum().item())
            for i in range(num_classes)
        ]
        llm_acc_epoch = (all_llm_preds == all_targets).float().mean().item()
        retrieval_acc_epoch = (all_retrieval_preds == all_targets).float().mean().item()
        per_class_str = ", ".join(
            f"{config.label_list[i]}: {per_class_acc[i]:.3f}"
            for i in range(num_classes)
        )
        fusion.train()
        retrieval_encoder.train()

        is_better = (accuracy > best_accuracy) or (
            np.isclose(accuracy, best_accuracy) and avg_loss < best_loss
        )
        if is_better:
            best_accuracy = accuracy
            best_loss = avg_loss
            best_epoch = epoch + 1
            best_state = {
                "retrieval_encoder": {
                    k: v.detach().cpu().clone()
                    for k, v in retrieval_encoder.state_dict().items()
                },
                "fusion": {
                    k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()
                },
                "beta": float(fusion.beta.detach().cpu().item()),
            }
            logger.info(
                f"New best epoch: {best_epoch} (acc={best_accuracy:.4f}, loss={best_loss:.4f})"
            )

        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch + 1}/{config.epochs} - loss: {avg_loss:.4f} - retrieval_loss: {avg_retrieval_aux_loss:.4f} - acc: {accuracy:.4f} - llm_acc: {llm_acc_epoch:.4f} - retrieval_acc: {retrieval_acc_epoch:.4f} - β: {beta_val:.4f} - lr: {current_lr:.2e} - per_class: [{per_class_str}]"
        )

        history["loss"].append(float(avg_loss))
        history["retrieval_aux_loss"].append(float(avg_retrieval_aux_loss))
        history["acc"].append(float(accuracy))
        history["llm_acc"].append(float(llm_acc_epoch))
        history["retrieval_acc"].append(float(retrieval_acc_epoch))
        history["beta"].append(float(beta_val))
        history["lr"].append(float(current_lr))
        history["per_class"].append([float(x) for x in per_class_acc])

        scheduler.step()

    state_to_save = {
        "retrieval_encoder": retrieval_encoder.state_dict(),
        "fusion": fusion.state_dict(),
        "beta": fusion.beta.item(),
    }
    if config.save_best_checkpoint and best_state is not None:
        logger.info(
            f"Saving best checkpoint from epoch {best_epoch}/{config.epochs} "
            f"(acc={best_accuracy:.4f}, loss={best_loss:.4f})"
        )
        state_to_save = best_state

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "retrieval_encoder": state_to_save["retrieval_encoder"],
            "fusion": state_to_save["fusion"],
            "beta": state_to_save["beta"],
            "config": {
                "model_name": config.model_name,
                "retriever_model": config.retriever_model,
                "top_k": config.top_k,
                "num_classes": num_classes,
                "initial_beta": config.initial_beta,
                "lambda_reg": config.lambda_reg,
                "evidence_mode": config.evidence_mode,
                "label_list": config.label_list,
                "normalize_branch_logits": bool(config.normalize_branch_logits),
                "adaptive_beta": bool(config.adaptive_beta),
                "retrieval_aux_loss_weight": float(config.retrieval_aux_loss_weight),
                "save_best_checkpoint": bool(config.save_best_checkpoint),
                "score_features": score_features,
                "nli_model": config.nli_model if config.use_nli else None,
                "interaction_dim": interaction_dim,
            },
        },
        save_path,
    )

    logger.info(f"Fusion model saved to {save_path}")

    _save_training_curves(
        history,
        save_path=save_path,
        best_epoch=best_epoch,
    )

    return save_path
