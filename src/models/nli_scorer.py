from __future__ import annotations

from typing import List, Optional

import numpy as np
from loguru import logger

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class NLIScorer:
    """Claim-evidence stance detection via multilingual NLI.

    Default model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
    (~278 MB, supports Vietnamese, trained on 2.7M multilingual NLI samples).

    Returns per-pair probabilities ordered as [entailment, neutral, contradiction]:
      - High entailment    → evidence supports the claim   → TRUE signal
      - High contradiction → evidence refutes the claim    → FALSE signal
      - High neutral       → evidence is unrelated         → NEI signal
    """

    DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 64,
        max_length: int = 512,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._ent_idx = 0
        self._neu_idx = 1
        self._con_idx = 2

    def load(self) -> "NLIScorer":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and transformers are required for NLIScorer.")
        logger.info(f"Loading NLI model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device).eval()

        id2label = {k: v.lower() for k, v in self._model.config.id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        self._ent_idx = label2id.get("entailment", 0)
        self._neu_idx = label2id.get("neutral", 1)
        self._con_idx = label2id.get("contradiction", 2)
        logger.info(
            f"NLI loaded | ent={self._ent_idx}, neu={self._neu_idx}, con={self._con_idx}"
        )
        return self

    def score(self, premises: List[str], hypotheses: List[str]) -> np.ndarray:
        """Score (premise=evidence, hypothesis=claim) pairs.

        Returns:
            np.ndarray of shape [N, 3]: columns = [entailment, neutral, contradiction]
        """
        if self._model is None:
            self.load()
        if not premises:
            return np.empty((0, 3), dtype=np.float32)

        all_probs: List[np.ndarray] = []
        for i in range(0, len(premises), self.batch_size):
            bp = premises[i : i + self.batch_size]
            bh = hypotheses[i : i + self.batch_size]
            inputs = self._tokenizer(
                bp,
                bh,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            ordered = np.stack(
                [probs[:, self._ent_idx], probs[:, self._neu_idx], probs[:, self._con_idx]],
                axis=-1,
            )
            all_probs.append(ordered)
        return np.concatenate(all_probs, axis=0)  # [N, 3]

    def unload(self) -> None:
        import gc

        del self._model, self._tokenizer
        self._model = self._tokenizer = None
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("NLI model unloaded.")
