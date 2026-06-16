import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, using numpy-based similarity")

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available, using TF-IDF")
    CrossEncoder = None

import re


try:
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer

    NLTK_AVAILABLE = True
except ImportError as e:
    NLTK_AVAILABLE = False
    logger.warning(f"NLTK not available, advanced preprocessing disabled. Error: {e}")


class TextPreprocessor:
    """
    Handles text preprocessing for retrieval:
    - URL removal
    - Stopword removal
    - Lemmatization
    """

    def __init__(self):
        if not NLTK_AVAILABLE:
            return

        # Download necessary NLTK data
        try:
            nltk.data.find("corpora/stopwords")
            nltk.data.find("corpora/wordnet")
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("taggers/averaged_perceptron_tagger")
        except LookupError:
            logger.info("Downloading NLTK resources...")
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def _get_wordnet_pos(self, treebank_tag):
        """Map NLTK POS tag to WordNet POS tag"""
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess text and return list of tokens.
        1. Remove URLs
        2. Lowercase
        3. Tokenize
        4. Remove stopwords & Lemmatize (with POS tags)
        """
        if not NLTK_AVAILABLE:
            # Fallback to simple tokenization
            return re.findall(r"\b\w+\b", text.lower())

        # 1. Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # 2. Lowercase
        text = text.lower()

        # 3. Tokenize
        try:
            tokens = nltk.word_tokenize(text)
        except LookupError:
            # Fallback if punkt fails
            tokens = re.findall(r"\b\w+\b", text)

        # 4. Remove stopwords and Lemmatize
        clean_tokens = []

        # Get POS tags for better lemmatization
        try:
            pos_tags = nltk.pos_tag(tokens)
        except LookupError:
            # Fallback if tagger fails
            pos_tags = [(t, "N") for t in tokens]

        for token, tag in pos_tags:
            # Simple check for alphanumeric to avoid punctuation
            if token.isalnum() and token not in self.stop_words:
                wn_tag = self._get_wordnet_pos(tag)
                clean_tokens.append(self.lemmatizer.lemmatize(token, wn_tag))

        return clean_tokens


@dataclass
class RetrievalResult:
    document_id: str
    text: str
    score: float
    rrf_score: float  # Changed from bm25_score to rrf_score
    recency_score: float
    cyclicity_score: float
    cosine_similarity: float  # Raw FAISS inner-product (cosine sim after L2 norm)
    timestamp: datetime
    metadata: Dict
    embedding: Optional[np.ndarray] = None  # L2-normalised BGE embedding of this document


class TemporalScorer:
    """
    Implements temporal scoring from Equation (1):
    Score(q, di) = α · BM25(q, di) + (1 − α) · Recency(di)

    With extensions from Section 3.5:
    - Cycle-Aware Scoring (Eq. 8)
    - Parameter Adaptation (Eq. 9)
    """

    def __init__(
        self,
        alpha: float = 0.7,
        lambda_decay: float = 0.1,
        gamma: float = 0.5,
        reference_date: datetime = None,  # Will be set dynamically per query if None
    ):
        """
        Initialize temporal scorer with paper parameters.

        Args:
            alpha: Weight for BM25 vs temporal (paper: 0.7)
            lambda_decay: Exponential decay factor (paper: 0.1)
            gamma: Recency vs cyclicity mix (paper: 0.5)
            reference_date: Reference date for recency calculation
        """
        self.alpha = alpha
        self.lambda_decay = lambda_decay
        self.lambda_base = lambda_decay
        self.gamma = gamma
        self._reference_date = reference_date  # Private, can be overridden

        logger.info(
            f"TemporalScorer initialized: α={alpha}, λ={lambda_decay}, γ={gamma}"
        )

    @property
    def reference_date(self) -> datetime:
        """Get reference date, defaulting to current time if not set."""
        if self._reference_date is None:
            return datetime.now(timezone.utc)
        return self._reference_date

    @reference_date.setter
    def reference_date(self, value: datetime):
        """Set reference date for recency calculations."""
        self._reference_date = value

    def calculate_recency(self, timestamp: datetime) -> float:
        """
        Calculate recency score using exponential decay: e^(-λt)

        Args:
            timestamp: Document timestamp

        Returns:
            Recency score between 0 and 1
        """
        if timestamp is None:
            return 0.0

        # Normalize timezone awareness
        ts = timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ref = self.reference_date
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)

        # Paper Eq.1: e^(-λt) where t is time difference
        # Use total_seconds for precise resolution (not quantized by .days)
        time_diff_seconds = (ref - ts).total_seconds()
        time_diff_days = max(0, time_diff_seconds / 86400.0)  # Convert to days
        return float(np.exp(-self.lambda_decay * time_diff_days))

    def calculate_cyclicity(
        self, timestamps: List[datetime], pattern_type: str = None
    ) -> float:
        """
        Calculate cyclicity score using Autocorrelation Function (ACF) to detect
        repeating lag patterns in document frequency over time.
        Implements Section 4.1's cycle-aware scoring (Eq.8).

        Under the null hypothesis that organic news follows a non-periodic Poisson
        process, ACF values at all lags are ~N(0, 1/N). Lags exceeding the 95%
        confidence bound (±1.96/√N) indicate coordinated/non-organic activity.

        Args:
            timestamps: List of document timestamps for pattern analysis
            pattern_type: Optional pattern type hint

        Returns:
            Cyclicity score between 0 and 1
        """
        valid_timestamps = [t for t in timestamps if t is not None]
        if len(valid_timestamps) < 10:
            return 0.5

        min_date = min(valid_timestamps)
        max_date = max(valid_timestamps)
        date_range = (max_date - min_date).days + 1

        if date_range < 14:
            return 0.5

        daily_counts = np.zeros(date_range)
        for ts in valid_timestamps:
            day_idx = (ts - min_date).days
            if 0 <= day_idx < date_range:
                daily_counts[day_idx] += 1

        # Mean-centre before computing ACF (required for unbiased estimate)
        x = daily_counts - daily_counts.mean()
        n = len(x)
        variance = np.dot(x, x)
        if variance < 1e-10:
            return 0.3

        # Compute normalised ACF via full convolution; keep lags 1..max_lag
        max_lag = min(n // 2, 90)
        acf = np.correlate(x, x, mode="full")[n - 1:]  # lags 0,1,2,...
        acf = acf[1 : max_lag + 1] / variance          # normalise; skip lag-0

        # 95 % confidence bound for white-noise null (Box & Jenkins, 1976)
        confidence = 1.96 / np.sqrt(n)

        significant = np.abs(acf) > confidence
        if not significant.any():
            return 0.3

        # Score = mean excess above threshold at significant lags, capped at 1
        excess = np.abs(acf[significant]) - confidence
        cyclicity = float(min(1.0, excess.mean() / confidence))
        return cyclicity

    def adapt_lambda(self, trend_indicator: float) -> None:
        """
        Dynamically adjust λ based on trend.
        Implements Equation (9): λt = Sigmoid(Trend(di)) · λbase

        Args:
            trend_indicator: Trend value between -1 and 1
        """
        sigmoid = 1 / (1 + np.exp(-trend_indicator))
        self.lambda_decay = sigmoid * self.lambda_base

    def calculate_trend(self, timestamps: List[datetime]) -> float:
        """
        Calculate trend indicator for adaptive λ.
        Simple approach: slope of recent occurrences.
        Returns value suitable for sigmoid (-3 to 3 range).
        """
        valid_timestamps = [t for t in timestamps if t is not None]
        if len(valid_timestamps) < 5:
            return 0.0

        # Sort and get recent window
        sorted_ts = sorted(valid_timestamps, reverse=True)[:30]
        if len(sorted_ts) < 5:
            return 0.0

        # Normalize timezone
        now = self.reference_date
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
            
        norm_ts = []
        for t in sorted_ts:
            if t.tzinfo is None:
                norm_ts.append(t.replace(tzinfo=timezone.utc))
            else:
                norm_ts.append(t)

        week1 = sum(1 for t in norm_ts if (now - t).days <= 7)
        week2 = sum(1 for t in norm_ts if 7 < (now - t).days <= 14)
        week3 = sum(1 for t in norm_ts if 14 < (now - t).days <= 21)

        # Trend: positive if increasing, negative if decreasing
        if week3 > 0:
            trend = (week1 - week3) / (week3 + 1)  # Normalized change
        else:
            trend = week1 / 3.0

        return np.clip(trend, -3, 3)  # For sigmoid input

    def calculate_temporal_score(
        self,
        timestamp: datetime,
        historical_timestamps: List[datetime] = None,
        use_adaptive_lambda: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Calculate combined temporal score.
        Implements Equation (8): temporal = γ·Recency + (1-γ)·Cyclicity
        with Equation (9): λt = Sigmoid(Trend) · λbase for adaptive decay

        Returns:
            Tuple of (temporal_score, recency_score, cyclicity_score)
        """
        # Apply adaptive lambda if historical data available
        if use_adaptive_lambda and historical_timestamps:
            trend = self.calculate_trend(historical_timestamps)
            self.adapt_lambda(trend)

        recency = self.calculate_recency(timestamp)

        if historical_timestamps:
            cyclicity = self.calculate_cyclicity(historical_timestamps)
        else:
            cyclicity = 0.5

        temporal_score = self.gamma * recency + (1 - self.gamma) * cyclicity

        # Reset lambda to base for next calculation
        if use_adaptive_lambda:
            self.lambda_decay = self.lambda_base

        return temporal_score, recency, cyclicity


import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Implements query expansion for Vietnamese banking & financial fact-checking.
    Uses domain-specific synonyms for better evidence retrieval.
    """

    def __init__(self, glossary: Dict[str, List[str]] = None):
        """
        Initialize query expander.

        Args:
            glossary: Dictionary mapping financial/banking terms to related synonyms
        """
        self.glossary = glossary or self._default_glossary()
        logger.info(f"QueryExpander initialized with {len(self.glossary)} terms")

    def _default_glossary(self) -> Dict[str, List[str]]:
        """Rich glossary for Vietnamese financial fact-checking"""

        return {
            # ==========================================
            # 1. FACT-CHECKING LABELS & ACTIONS
            # ==========================================
            "đúng": [
                "chính xác",
                "xác thực",
                "được xác nhận",
                "đúng sự thật",
                "chuẩn xác",
            ],
            "sai": [
                "không chính xác",
                "thông tin sai",
                "sai lệch",
                "không đúng",
                "bác bỏ",
            ],
            "thiếu thông tin": [
                "chưa đầy đủ",
                "thiếu ngữ cảnh",
                "không đủ dữ kiện",
                "một phần sự thật",
            ],
            "bằng chứng": [
                "nguồn chứng minh",
                "tài liệu",
                "dữ liệu xác thực",
                "nguồn tham khảo",
                "văn bản gốc",
            ],
            "xác minh": ["kiểm chứng", "đối chiếu", "thẩm định", "kiểm tra", "làm rõ"],
            "tin giả": [
                "fake news",
                "tin bịa đặt",
                "thông tin thất thiệt",
                "tin đồn thất thiệt",
            ],
            "cảnh báo": ["warning", "khuyến cáo", "báo động", "lưu ý"],
            # ==========================================
            # 2. CORE BANKING & INSTITUTIONS
            # ==========================================
            "ngân hàng": ["tổ chức tín dụng", "bank", "ngân hàng thương mại", "NHTM"],
            "ngân hàng nhà nước": [
                "SBV",
                "NHNN",
                "central bank",
                "ngân hàng trung ương",
            ],
            "chi nhánh": ["branch", "phòng giao dịch", "PGD", "điểm giao dịch"],
            # ==========================================
            # 3. MACROECONOMICS & POLICIES
            # ==========================================
            "lãi suất": [
                "interest rate",
                "lãi vay",
                "lãi tiền gửi",
                "lãi suất điều hành",
            ],
            "lạm phát": [
                "inflation",
                "CPI",
                "chỉ số giá tiêu dùng",
                "mất giá đồng tiền",
            ],
            "tỷ giá": [
                "exchange rate",
                "USD/VND",
                "ngoại tệ",
                "thị trường ngoại hối",
                "forex",
            ],
            "chính sách tiền tệ": [
                "monetary policy",
                "nới lỏng tiền tệ",
                "thắt chặt tiền tệ",
                "bơm hút tiền",
            ],
            "room tín dụng": [
                "hạn mức tín dụng",
                "credit quota",
                "chỉ tiêu tăng trưởng tín dụng",
            ],
            "dự trữ bắt buộc": [
                "reserve requirement",
                "tỷ lệ dự trữ",
                "tiền gửi bắt buộc",
            ],
            # ==========================================
            # 4. BANKING PRODUCTS & OPERATIONS
            # ==========================================
            "tín dụng": ["credit", "khoản vay", "cho vay", "cấp vốn"],
            "tiền gửi": [
                "deposit",
                "savings",
                "gửi tiết kiệm",
                "tiền gửi thanh toán",
                "CASA",
            ],
            "thẻ tín dụng": [
                "credit card",
                "thẻ ghi nợ",
                "visa",
                "mastercard",
                "thẻ ngân hàng",
            ],
            "vay thế chấp": [
                "mortgage",
                "vay có tài sản đảm bảo",
                "thế chấp tài sản",
                "cầm cố",
            ],
            "vay tín chấp": [
                "unsecured loan",
                "vay không tài sản đảm bảo",
                "vay tiêu dùng",
            ],
            "giải ngân": ["disbursement", "rót vốn", "chuyển tiền vay"],
            "đáo hạn": ["maturity", "tất toán", "đến hạn thanh toán", "gia hạn nợ"],
            "chuyển khoản": [
                "bank transfer",
                "chuyển tiền",
                "remittance",
                "chuyển mạch",
            ],
            # ==========================================
            # 5. FINANCIAL METRICS & HEALTH
            # ==========================================
            "thanh khoản": [
                "liquidity",
                "dòng tiền",
                "khả năng chi trả",
                "khả năng thanh toán",
            ],
            "nợ xấu": [
                "bad debt",
                "non-performing loan",
                "NPL",
                "nợ nhóm 3",
                "nợ quá hạn",
            ],
            "biên lãi thuần": ["NIM", "net interest margin", "chênh lệch lãi suất"],
            "vốn điều lệ": ["charter capital", "vốn chủ sở hữu", "quy mô vốn", "CAR"],
            # ==========================================
            # 6. FINANCIAL MARKETS
            # ==========================================
            "trái phiếu": [
                "bond",
                "trái phiếu doanh nghiệp",
                "debt securities",
                "TPDN",
                "trái phiếu chính phủ",
            ],
            "cổ phiếu": ["stock", "shares", "mã chứng khoán", "cổ phần"],
            "chứng khoán": [
                "securities",
                "thị trường vốn",
                "VN-Index",
                "sàn giao dịch",
            ],
            "bảo hiểm": [
                "insurance",
                "bancassurance",
                "bảo hiểm nhân thọ",
                "hợp đồng bảo hiểm",
            ],
            # ==========================================
            # 7. FRAUD, RISK & CYBERSECURITY
            # ==========================================
            "lừa đảo": ["scam", "gian lận", "chiếm đoạt", "mạo danh", "giả danh"],
            "rút tiền hàng loạt": [
                "bank run",
                "withdrawal panic",
                "khủng hoảng thanh khoản",
                "ồ ạt rút tiền",
            ],
            "phá sản": ["bankrupt", "mất khả năng thanh toán", "sụp đổ", "vỡ nợ"],
            "chiếm đoạt tài khoản": [
                "account takeover",
                "mất tiền trong thẻ",
                "bị hack tài khoản",
                "trừ tiền vô lý",
            ],
            "đường link lạ": [
                "phishing link",
                "link giả mạo",
                "trang web lừa đảo",
                "web đen",
            ],
            "mã độc": [
                "malware",
                "virus",
                "phần mềm gián điệp",
                "app giả mạo",
                "ứng dụng độc hại",
            ],
            "rửa tiền": [
                "money laundering",
                "AML",
                "hợp pháp hóa tiền bẩn",
                "nguồn tiền bất hợp pháp",
            ],
        }

    def expand_query(self, query: str) -> str:
        query_lower = query.lower()
        expanded_terms = [query]

        for term, synonyms in self.glossary.items():
            if term in query_lower:
                # Add first two synonyms to avoid over-expanding the query
                expanded_terms.extend(synonyms[:2])

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for word in expanded_terms:
            if word not in seen:
                seen.add(word)
                result.append(word)

        return " ".join(result)


class KnowledgeAugmentedRetriever:
    """
    Main retrieval system implementing hybrid RRF-based retrieval.

    Combines:
    - BGE embeddings for semantic search (via FAISS) - Stage 1a
    - BM25 for lexical matching - Stage 1b
    - Reciprocal Rank Fusion (RRF) to combine rankings - Stage 2
    - Temporal scoring for recency awareness - Stage 3
    """

    def __init__(
        self,
        embedding_model: str = "AITeamVN/Vietnamese_Embedding",
        alpha: float = 0.7,
        lambda_decay: float = 0.1,
        gamma: float = 0.5,
        use_query_expansion: bool = True,
        rrf_k: int = 60,
        index_path: str = None,
    ):
        """
        Initialize the retrieval system.

        Args:
            embedding_model: Name of sentence transformer model
            alpha: RRF vs temporal weight (default: 0.7)
            lambda_decay: Recency decay factor (default: 0.1)
            gamma: Recency vs cyclicity mix (default: 0.5)
            use_query_expansion: Enable query expansion
            rrf_k: RRF constant k (default: 60)
            index_path: Path to save/load FAISS index
        """
        self.alpha = alpha
        self.index_path = index_path
        self.rrf_k = rrf_k

        # Initialize components
        self.temporal_scorer = TemporalScorer(
            alpha=alpha, lambda_decay=lambda_decay, gamma=gamma
        )

        self.query_expander = QueryExpander() if use_query_expansion else None
        self.preprocessor = TextPreprocessor()

        # Initialize embedding model (Bi-Encoder for FAISS)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info(f"Loading embedding model: {embedding_model}")
            self.encoder = SentenceTransformer(embedding_model)
            self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        else:
            self.encoder = None
            import os

            self.embedding_dim = int(
                os.getenv("RETRIEVER_EMBEDDING_DIM", "1024")
            )  # Use from env var

        # Cross-Encoder removed - using RRF-based hybrid retrieval instead

        # Storage
        self.documents = []
        self.document_embeddings = None
        self.bm25 = None
        self.faiss_index = None

        logger.info("KnowledgeAugmentedRetriever initialized")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenization using preprocessor"""
        return self.preprocessor.preprocess(text)

    def index_documents(
        self,
        documents: List[Dict],
        text_field: str = "text",
        id_field: str = "id",
        timestamp_field: str = "timestamp",
    ) -> None:
        """
        Index documents for retrieval.

        Args:
            documents: List of document dictionaries
            text_field: Key for document text
            id_field: Key for document ID
            timestamp_field: Key for timestamp
        """
        logger.info(f"Indexing {len(documents)} documents...")

        self.documents = []
        texts = []

        for doc in documents:
            doc_entry = {
                "id": doc.get(id_field, str(len(self.documents))),
                "text": doc.get(text_field, ""),
                "timestamp": doc.get(timestamp_field, datetime.now()),
                "metadata": {
                    k: v
                    for k, v in doc.items()
                    if k not in [text_field, id_field, timestamp_field]
                },
            }
            self.documents.append(doc_entry)
            texts.append(doc_entry["text"])

        # Build BM25 index
        # Note: Using BM25Okapi (standard BM25). Paper mentions BM25+ in Eq.8.
        # BM25+ adds delta term to avoid negative scores, but difference is minor.
        tokenized = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized)

        # Build FAISS index
        if self.encoder is not None:
            logger.info("Creating embeddings...")
            self.document_embeddings = self.encoder.encode(
                texts, show_progress_bar=True, convert_to_numpy=True
            )

            if FAISS_AVAILABLE:
                # Create FAISS index
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                # Normalize for cosine similarity
                faiss.normalize_L2(self.document_embeddings)
                self.faiss_index.add(self.document_embeddings)
                logger.info(
                    f"FAISS index created with {self.faiss_index.ntotal} vectors"
                )

        logger.info(f"Indexing complete: {len(self.documents)} documents")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_temporal: bool = True,
        expand_query: bool = True,
        use_semantic: bool = True,
        rrf_top_k: int = 20,
    ) -> List[RetrievalResult]:

        if not self.documents:
            logger.warning("No documents indexed")
            return []

        # Update reference date to current query time
        self.temporal_scorer.reference_date = datetime.now(timezone.utc)

        # Query expansion
        if expand_query and self.query_expander:
            expanded_query = self.query_expander.expand_query(query)
        else:
            expanded_query = query

        # Stage 1a: BM25 scoring for all documents
        tokenized_query = self._tokenize(expanded_query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Create BM25 rankings (higher score = lower rank number)
        bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
        bm25_ranks = {idx: rank for rank, (idx, score) in enumerate(bm25_ranked)}

        # Stage 1b: FAISS semantic search (if available)
        faiss_ranks = {}
        faiss_cosine = {}  # idx → cosine similarity (inner product after L2 norm)
        if use_semantic and self.encoder is not None and self.faiss_index is not None:
            query_embedding = self.encoder.encode(
                [expanded_query], convert_to_numpy=True
            )
            faiss.normalize_L2(query_embedding)
            # Store L2-normalised query embedding for claim-evidence interaction features
            self._last_query_embedding = query_embedding[0].copy()

            # Search all documents to get complete ranking
            n_docs = len(self.documents)
            distances, indices = self.faiss_index.search(query_embedding, n_docs)
            faiss_ranks = {idx: rank for rank, idx in enumerate(indices[0].tolist())}
            faiss_cosine = {int(indices[0][i]): float(distances[0][i]) for i in range(len(indices[0]))}
        else:
            self._last_query_embedding = None
            # No FAISS available: use uniform ranks
            faiss_ranks = {i: i for i in range(len(self.documents))}

        # Stage 2: Reciprocal Rank Fusion (RRF)
        # RRF(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_dense(d))
        rrf_scores = {}
        for i in range(len(self.documents)):
            bm25_rank = bm25_ranks.get(i, len(self.documents))
            faiss_rank = faiss_ranks.get(i, len(self.documents))
            rrf_scores[i] = (1.0 / (self.rrf_k + bm25_rank)) + (
                1.0 / (self.rrf_k + faiss_rank)
            )

        # Sort by RRF score and take top rrf_top_k (default 20)
        rrf_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        rrf_candidates = rrf_ranked[:rrf_top_k]

        # Normalize RRF scores to [0,1] for final score calculation
        max_rrf = max(score for idx, score in rrf_candidates) if rrf_candidates else 1.0
        rrf_scores_norm = {idx: score / max_rrf for idx, score in rrf_candidates}

        # Build group histories from FULL corpus for cyclicity calculation
        docs_by_group = {}
        for i, doc in enumerate(self.documents):
            group_key = (
                doc["metadata"].get("type")
                or doc["metadata"].get("source")
                or "default"
            )
            if group_key not in docs_by_group:
                docs_by_group[group_key] = []
            docs_by_group[group_key].append((i, doc["timestamp"]))

        # Stage 3: Temporal scoring on RRF candidates
        final_scores = []

        for idx, rrf_score in rrf_candidates:
            doc = self.documents[idx]

            if use_temporal:
                # Get group-specific timestamps for cyclicity
                group_key = (
                    doc["metadata"].get("type")
                    or doc["metadata"].get("source")
                    or "default"
                )
                group_timestamps = [
                    ts for idx_g, ts in docs_by_group.get(group_key, [])
                ]

                # Calculate temporal score with group-based cyclicity + adaptive λ
                temporal, recency, cyclicity = (
                    self.temporal_scorer.calculate_temporal_score(
                        doc["timestamp"], group_timestamps, use_adaptive_lambda=True
                    )
                )
                # Final Score = α × RRF + (1-α) × Temporal
                score = self.alpha * rrf_scores_norm[idx] + (1 - self.alpha) * temporal
            else:
                recency, cyclicity = 0.5, 0.5
                score = rrf_scores_norm[idx]

            final_scores.append(
                {
                    "index": idx,
                    "score": score,
                    "rrf_score": rrf_scores_norm[idx],
                    "recency_score": recency,
                    "cyclicity_score": cyclicity,
                    "cosine_similarity": faiss_cosine.get(idx, 0.0),
                }
            )

        # Sort by final score and take top_k (default 10)
        final_scores.sort(key=lambda x: x["score"], reverse=True)
        top_results = final_scores[:top_k]

        # Build final results
        results = []
        for item in top_results:
            doc = self.documents[item["index"]]
            doc_emb = (
                self.document_embeddings[item["index"]]
                if self.document_embeddings is not None
                else None
            )
            results.append(
                RetrievalResult(
                    document_id=doc["id"],
                    text=doc["text"],
                    score=item["score"],
                    rrf_score=item["rrf_score"],  # Changed from bm25_score
                    recency_score=item["recency_score"],
                    cyclicity_score=item["cyclicity_score"],
                    cosine_similarity=item["cosine_similarity"],
                    timestamp=doc["timestamp"],
                    metadata=doc["metadata"],
                    embedding=doc_emb,
                )
            )

        return results

    def save_index(self, path: str = None) -> None:
        """Save the index to disk"""
        path = path or self.index_path
        if path is None:
            logger.warning("No path specified for saving index")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            "documents": self.documents,
            "document_embeddings": self.document_embeddings,
            "alpha": self.alpha,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Index saved to {path}")

    def load_index(self, path: str = None) -> None:
        """Load the index from disk"""
        path = path or self.index_path
        if path is None or not os.path.exists(path):
            logger.warning(f"Index file not found: {path}")
            return

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.documents = state["documents"]
        self.document_embeddings = state.get("document_embeddings")
        self.alpha = state.get("alpha", self.alpha)

        # Rebuild indices
        texts = [doc["text"] for doc in self.documents]
        tokenized = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized)

        if self.document_embeddings is not None and FAISS_AVAILABLE:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            # Normalize for cosine similarity (must match indexing behavior)
            faiss.normalize_L2(self.document_embeddings)
            self.faiss_index.add(self.document_embeddings)

        logger.info(f"Index loaded from {path}: {len(self.documents)} documents")


if __name__ == "__main__":
    print("KnowledgeAugmentedRetriever module. Use via pipeline.")
