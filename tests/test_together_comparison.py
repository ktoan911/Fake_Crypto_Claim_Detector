#!/usr/bin/env python3
"""
So sánh model LoRA / Fusion (fine-tune nội bộ) với các model general của
Together AI trên cùng tập dữ liệu (mặc định: data/time_dataset.json).

Ý tưởng:
- Together models được dùng như một fact-checker zero-shot: nạp cùng
  PROMPT_TEMPLATE (src/config.py) và bắt model xuất đúng 1 chữ cái A/B/C.
- LoRA / Fusion dùng lại pipeline trong test_all_modes.py (retrieval +
  LLMScorer + fusion layer).
- Cuối cùng in ra 1 bảng metric thống nhất để so sánh.

Cách gọi Together API mượn nguyên retry wrapper trong src/llm_call.py.

Ví dụ:
  # Chỉ chạy Together (không cần GPU) với gold evidence
  python tests/test_together_comparison.py --limit 50

  # Thêm nhiều model Together
  python tests/test_together_comparison.py \
      --together_models Qwen/Qwen3.6-Plus meta-llama/Llama-3.3-70B-Instruct-Turbo

  # Chạy kèm cả LoRA + Fusion để so sánh 3 bên (cần GPU + model)
  python tests/test_together_comparison.py --run_local --limit 50
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from loguru import logger
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import LABEL_LIST, PROMPT_TEMPLATE
from src.llm_call import _chat_completion_with_retry

# Reuse metric + fusion loader from the existing test script
from tests.test_all_modes import calculate_metrics, load_fusion_model

# ---------------------------------------------------------------------------
# Env-var defaults (mirror .env / test_all_modes.py)
# ---------------------------------------------------------------------------
_DEFAULT_LLM_MODEL = os.getenv("LLM_FINETUNE", "ktoan911/Qwen3-4B-factcheck-finetune-v6")
_DEFAULT_FUSION_MODEL = os.getenv("FUSION_MODEL", "ktoan911/fact-check-fusion-model")
_DEFAULT_RETRIEVER_MODEL = os.getenv("RETRIEVER_MODEL", "AITeamVN/Vietnamese_Embedding")

# Danh sách model Together mặc định để đối chứng (có thể override bằng --together_models)
_DEFAULT_TOGETHER_MODELS = [
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-V4-Pro",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
]


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------
def normalize_label_to_id(val) -> int:
    """Map nhãn dataset (đúng/sai/thiếu thông tin, hoặc A/B/C) -> id 0/1/2."""
    v = str(val).lower().strip()
    if v in ["a", "true", "đúng", "dung", "supported", "legit", "0"]:
        return 0
    if v in ["b", "false", "sai", "refuted", "contradicted", "scam", "fake", "1"]:
        return 1
    if v in [
        "c",
        "nei",
        "thieu",
        "thiếu",
        "thiếu thông tin",
        "thieu thong tin",
        "chưa chắc chắn",
        "not enough info",
        "insufficient",
        "unclear",
        "2",
    ]:
        return 2
    return 2  # default -> Insufficient/C


def parse_letter_prediction(text: str) -> int:
    """Trích A/B/C từ output của LLM -> id. Ưu tiên chữ cái đứng cuối
    (vì 'Conclusion:' nằm cuối prompt, model reasoning cũng chốt ở cuối)."""
    if not text:
        return 2
    matches = re.findall(r"\b([ABC])\b", text.strip())
    if matches:
        return {"A": 0, "B": 1, "C": 2}[matches[-1]]
    # Fallback: bắt theo từ khóa tiếng Việt nếu model không xuất chữ cái
    low = text.lower()
    if "thiếu" in low or "không đủ" in low or "chưa chắc" in low:
        return 2
    if "đúng" in low or "hỗ trợ" in low or "chính xác" in low:
        return 0
    if "sai" in low or "mâu thuẫn" in low or "bác bỏ" in low:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def format_evidence(evidence_list) -> str:
    """Ghép evidence thành chuỗi đánh số như LLMScorer nội bộ (1. ...\\n2. ...).

    Chỉ lấy `content`, KHÔNG chèn timestamp: LoRA và Together đều không được
    biết thời gian bằng chứng. Chỉ fusion model nhận thông tin thời gian, và
    qua feature recency (số) chứ không phải qua text.
    """
    parts = []
    for ev in evidence_list:
        if isinstance(ev, dict):
            content = str(ev.get("content", "")).strip()
        else:
            content = str(ev).strip()
        if content:
            parts.append(content)
    return "\n".join(f"{i + 1}. {p}" for i, p in enumerate(parts))


def load_dataset(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    if limit:
        data = data[:limit]
        logger.info(f"Limited to {limit} samples")

    texts = [d["claim"] for d in data]
    labels = [normalize_label_to_id(d["label"]) for d in data]
    # gold evidence text thuần — dùng chung cho cả Together và LoRA (không timestamp)
    gold_evidence_str = [format_evidence(d.get("evidence", [])) for d in data]
    raw_evidence = [d.get("evidence", []) for d in data]
    return texts, labels, gold_evidence_str, raw_evidence


# ---------------------------------------------------------------------------
# Together AI zero-shot evaluation
# ---------------------------------------------------------------------------
def eval_together_model(
    model_name,
    texts,
    gold_evidence_str,
    max_tokens=1024,
    max_workers=8,
):
    """Gọi 1 model Together cho toàn bộ claim (song song) -> list pred id."""
    logger.info(f"[Together] Evaluating model: {model_name}")

    def _predict_one(idx):
        prompt = PROMPT_TEMPLATE.format(
            claim=texts[idx], evidence=gold_evidence_str[idx]
        )
        try:
            resp = _chat_completion_with_retry(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            content = (resp.choices[0].message.content or "").strip()
            return idx, parse_letter_prediction(content)
        except Exception as e:
            logger.warning(f"[Together:{model_name}] sample {idx} failed: {e}")
            return idx, 2  # fallback -> C

    preds = [2] * len(texts)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_predict_one, i) for i in range(len(texts))]
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc=f"Together:{model_name}"
        ):
            idx, pred = fut.result()
            preds[idx] = pred
    return preds


# ---------------------------------------------------------------------------
# Local LoRA + Fusion pipeline (reuse test_all_modes logic)
# ---------------------------------------------------------------------------
def run_local_pipeline(args, texts, labels, gold_evidence_str, raw_evidence):
    """Chạy LoRA + Fusion (gold & retrieval). Trả về dict {mode_name: preds}."""
    import torch

    from src.llm_scorer import LLMScorer
    from src.retrieval.retrieval import KnowledgeAugmentedRetriever
    from src.training.fusion_trainer import _build_retrieval_features
    from src.utils import normalize_text as _norm

    device = args.device
    logger.info(f"[Local] device={device}")

    # Fusion model + config
    retrieval_encoder, fusion_layer, fusion_config = load_fusion_model(
        args.fusion_model, device
    )
    top_k = fusion_config.get("top_k", 10)
    retriever_model = (
        args.retriever_model
        or fusion_config.get("retriever_model")
        or _DEFAULT_RETRIEVER_MODEL
    )
    dynamic_label_list = fusion_config.get("label_list", LABEL_LIST)

    # Build KB từ evidence của test set (dedup)
    unique_docs = {}
    for evs in raw_evidence:
        for ev in evs:
            content = ev.get("content", "") if isinstance(ev, dict) else str(ev)
            content = content.strip()
            if len(content) > 10:
                key = _norm(content)
                if key not in unique_docs:
                    ts = ev.get("timestamp") if isinstance(ev, dict) else None
                    unique_docs[key] = {"text": content, "timestamp": ts}
    kb_docs = list(unique_docs.values())
    logger.info(f"[Local] KB with {len(kb_docs)} unique docs")

    retriever = KnowledgeAugmentedRetriever(embedding_model=retriever_model, rrf_k=60)
    retriever.index_documents(kb_docs, text_field="text", timestamp_field="timestamp")

    llm = LLMScorer(
        model_name=args.lora_model,
        device=device,
        max_length=int(os.getenv("LLM_MAX_LENGTH", "2048")),
        labels=dynamic_label_list,
        prompt_template=PROMPT_TEMPLATE,
    )

    # Retrieval features
    logger.info("[Local] Retrieving...")
    all_base, all_inter, all_retrieved = [], [], []
    for text in tqdm(texts, desc="Retrieving"):
        feats, interaction, retrieved = _build_retrieval_features(retriever, text, top_k)
        all_base.append(feats)
        all_inter.append(interaction)
        all_retrieved.append(retrieved)

    # NLI features nếu checkpoint train kèm NLI
    all_features = all_base
    nli_model_name = fusion_config.get("nli_model") or None
    if nli_model_name:
        import gc as _gc

        from src.models.nli_scorer import NLIScorer

        logger.info(f"[Local] NLI scoring with {nli_model_name}...")
        _nli = NLIScorer(model_name=nli_model_name, device=device).load()
        flat_docs, flat_claims = [], []
        for text, evs in zip(texts, all_retrieved):
            for doc in evs:
                flat_docs.append(doc)
                flat_claims.append(text)
        nli_flat = _nli.score(premises=flat_docs, hypotheses=flat_claims)
        _nli.unload()
        del _nli
        _gc.collect()

        cursor = 0
        all_features = []
        for base_feats, evs in zip(all_base, all_retrieved):
            n_real = len(evs)
            nli_padded = np.full((top_k, 3), 1.0 / 3.0, dtype=np.float32)
            if n_real > 0:
                nli_padded[:n_real] = nli_flat[cursor : cursor + n_real]
            cursor += n_real
            all_features.append(np.concatenate([base_feats, nli_padded], axis=-1))

    tensor_features = torch.tensor(np.array(all_features), dtype=torch.float32).to(device)
    tensor_inter = None
    _non_none = [x for x in all_inter if x is not None]
    if _non_none:
        emb_shape = _non_none[0].shape[0]
        _filled = [
            x if x is not None else np.zeros(emb_shape, dtype=np.float32)
            for x in all_inter
        ]
        tensor_inter = torch.tensor(np.array(_filled, dtype=np.float32)).to(device)

    # LLM logits — chỉ chế độ GOLD (LLM đọc gold evidence).
    # Retrieval vẫn chạy ở trên nhưng chỉ để tạo feature cho fusion.
    logger.info("[Local] LLM inference (gold only)...")
    llm_bs = int(os.getenv("LLM_INFER_BATCH_SIZE", "16"))
    logits_gold = []
    for i in tqdm(range(0, len(texts), llm_bs), desc="LLM"):
        bt = texts[i : i + llm_bs]
        bg = gold_evidence_str[i : i + llm_bs]
        logits_gold.append(llm.score_logits(bt, bg))
    tensor_logits_gold = torch.cat(logits_gold, dim=0).to(device)

    results = {}
    results["LoRA + Gold"] = torch.argmax(tensor_logits_gold, dim=1).cpu().numpy()

    with torch.no_grad():
        encoded = retrieval_encoder(tensor_features, tensor_inter)
        out_gold = fusion_layer(tensor_logits_gold, encoded)
        results["Fusion + Gold"] = (
            torch.argmax(out_gold.final_probs, dim=1).cpu().numpy()
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="So sánh LoRA/Fusion với các model Together AI trên time_dataset.json"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "time_dataset.json",
        ),
        help="Đường dẫn dataset JSON (mặc định data/time_dataset.json)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--together_models",
        nargs="+",
        default=_DEFAULT_TOGETHER_MODELS,
        help="Danh sách model Together AI cần đối chứng",
    )
    parser.add_argument(
        "--skip_together", action="store_true", help="Bỏ qua phần Together AI"
    )
    parser.add_argument("--together_max_tokens", type=int, default=1024)
    parser.add_argument("--together_workers", type=int, default=8)
    # Local LoRA + Fusion
    parser.add_argument(
        "--run_local",
        action="store_true",
        help="Chạy kèm pipeline LoRA + Fusion (cần GPU + model)",
    )
    parser.add_argument("--lora_model", type=str, default=_DEFAULT_LLM_MODEL)
    parser.add_argument("--fusion_model", type=str, default=_DEFAULT_FUSION_MODEL)
    parser.add_argument("--retriever_model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Nếu set, ghi bảng kết quả ra CSV",
    )
    args = parser.parse_args()

    if args.device is None:
        try:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    # 1. Load data
    logger.info(f"Loading data from {args.data}...")
    texts, labels, gold_evidence_str, raw_evidence = load_dataset(args.data, args.limit)
    logger.info(f"Loaded {len(texts)} samples. Label dist: "
                f"A={labels.count(0)} B={labels.count(1)} C={labels.count(2)}")

    results_summary = []

    # 2. Together AI models (zero-shot, gold evidence — KHÔNG timestamp, như LoRA)
    if not args.skip_together:
        for model_name in args.together_models:
            model_name = str(model_name).strip()
            if not model_name:
                continue  # bỏ qua tên model rỗng
            preds = eval_together_model(
                model_name,
                texts,
                gold_evidence_str,
                max_tokens=args.together_max_tokens,
                max_workers=args.together_workers,
            )
            results_summary.append(
                calculate_metrics(
                    labels, preds, f"TAI:{model_name.split('/')[-1]}", LABEL_LIST
                )
            )

    # 3. Local LoRA + Fusion
    if args.run_local:
        local_results = run_local_pipeline(
            args, texts, labels, gold_evidence_str, raw_evidence
        )
        for mode_name, preds in local_results.items():
            results_summary.append(
                calculate_metrics(labels, preds, mode_name, LABEL_LIST)
            )

    if not results_summary:
        logger.warning("Không có model nào được đánh giá (dùng --run_local hoặc bỏ --skip_together).")
        return

    # 4. Summary table
    metric_cols = ["Acc", "Prec", "Rec", "F1_Mac", "F1_W"] + [
        f"F1_{l}" for l in LABEL_LIST
    ]
    print("\n" + "=" * 110)
    headers = [f"{'Model / Mode':<28}"] + [f"{h:<8}" for h in metric_cols]
    print(" | ".join(headers))
    print("-" * 110)
    key_map = {
        "Acc": "Accuracy",
        "Prec": "Precision",
        "Rec": "Recall",
        "F1_Mac": "F1_Macro",
        "F1_W": "F1_Weighted",
    }
    for res in results_summary:
        row = [f"{res['Mode']:<28}"]
        for col in metric_cols:
            key = key_map.get(col, col)
            row.append(f"{res[key]:<8.4f}")
        print(" | ".join(row))
    print("=" * 110 + "\n")

    if args.output_csv:
        import pandas as pd

        pd.DataFrame(results_summary).to_csv(args.output_csv, index=False)
        logger.info(f"Saved results to {args.output_csv}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
