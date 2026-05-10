from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple

import faiss
import gradio as gr
import numpy as np
import requests
print("RUNNING THIS FILE:", __file__)
# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("amazon_electronics_real_rag")


# ============================================================
# Configuration
# ============================================================
@dataclass(frozen=True)
class AppConfig:
    index_path: Path = Path(os.getenv("RAG_INDEX_PATH", r"C:\Users\pola\Desktop\RAG project\output_data\electronics_index1.faiss"))
    metadata_path: Path = Path(os.getenv("RAG_METADATA_PATH", r"C:\Users\pola\Desktop\RAG project\output_data\metadata_map1.jsonl"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf")
    chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
    request_timeout: int = int(os.getenv("RAG_REQUEST_TIMEOUT", "180"))
    max_retries: int = int(os.getenv("RAG_MAX_RETRIES", "3"))
    semantic_top_k: int = int(os.getenv("RAG_SEMANTIC_TOP_K", "24"))
    lexical_top_k: int = int(os.getenv("RAG_LEXICAL_TOP_K", "32"))
    candidate_pool: int = int(os.getenv("RAG_CANDIDATE_POOL", "24"))
    llm_shortlist_k: int = int(os.getenv("RAG_LLM_SHORTLIST_K", "6"))
    semantic_min_score: float = float(os.getenv("RAG_MIN_SEMANTIC_SCORE", "0.12"))
    default_browse_count: int = 3
    default_recommend_count: int = 2
    default_compare_count: int = 2
    max_feature_lines: int = 6

    @property
    def embed_url(self) -> str:
        return f"{self.ollama_base_url}/api/embed"

    @property
    def generate_url(self) -> str:
        return f"{self.ollama_base_url}/api/generate"


CONFIG = AppConfig()


# ============================================================
# Helpers
# ============================================================
UNICODE_REPLACEMENTS = {
    "\u00d7": "x",
    "\u03a9": "Ohm",
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "--",
    "\u2026": "...",
    "\u00b0": " degrees",
    "\u00b2": "2",
    "\u00b3": "3",
    "\u00bd": "1/2",
    "\u00bc": "1/4",
    "\u00be": "3/4",
    "\u2122": "",
    "\u00ae": "",
    "\u00a9": "",
}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
    "of", "to", "in", "on", "at", "for", "with", "by", "from", "into", "over", "under",
    "and", "or", "than", "that", "this", "these", "those", "my", "your", "their", "our",
    "show", "give", "tell", "find", "get", "need", "want", "looking", "look", "about",
    "me", "i", "it", "its", "please", "all", "some", "any", "product", "products",
    "item", "items", "thing", "things", "device", "devices",
}

GENERIC_QUERY_WORDS = {
    "best", "top", "good", "better", "overall", "quality", "value", "popular", "rated",
    "recommend", "recommendation", "compare", "comparison", "versus", "vs", "against",
    "price", "cheap", "cheapest", "expensive", "highest", "lowest", "most", "least",
}

ASIN_PATTERN = re.compile(r"\bB[A-Z0-9]{9}\b", flags=re.IGNORECASE)
COMPARE_PATTERN = re.compile(r"\b(?:vs\.?|versus|compare(?:\s+between)?|against)\b", flags=re.IGNORECASE)
COUNT_PATTERN = re.compile(r"\b(?:show|give|recommend|list)\s+(\d+)\b", flags=re.IGNORECASE)
EXACT_SPEC_PATTERN = re.compile(r"\b(hdmi|usb|bluetooth|wifi|wi-fi|pcie|usb-c|usbc|displayport|dp)\s*([0-9]+(?:\.[0-9]+)?)\b", flags=re.IGNORECASE)
MIN_SPEC_PATTERN = re.compile(
    r"\b(?:at least|or above|or higher|minimum|min(?:imum)?\s+of|supports?)\s+([a-z\-]+)?\s*([0-9]+(?:\.[0-9]+)?)\b",
    flags=re.IGNORECASE,
)

SORT_RELEVANCE = "relevance"
SORT_VALUE = "value"
SORT_RATING = "rating"
SORT_REVIEWS = "reviews"
SORT_PRICE_ASC = "price_asc"
SORT_PRICE_DESC = "price_desc"
SORT_INFO = "info"

MODE_LOOKUP = "lookup"
MODE_COMPARE = "compare"
MODE_RECOMMEND = "recommend"
MODE_BROWSE = "browse"

SPEC_EXACT = "exact"
SPEC_MINIMUM = "minimum"

LABEL_DIRECT = "direct_match"
LABEL_RELATED = "related_accessory"
LABEL_WEAK = "weak_match"
LABEL_UNRELATED = "unrelated"
VALID_LABELS = {LABEL_DIRECT, LABEL_RELATED, LABEL_WEAK, LABEL_UNRELATED}


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " | ".join(part for part in (safe_text(v) for v in value) if part)
    if isinstance(value, dict):
        parts: List[str] = []
        for key, item in value.items():
            text = safe_text(item)
            if text:
                parts.append(f"{key}: {text}")
        return " | ".join(parts)
    return str(value).strip()


def safe_float(value: Any) -> Optional[float]:
    try:
        text = safe_text(value)
        return float(text) if text != "" else None
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    try:
        text = safe_text(value)
        return int(float(text)) if text != "" else None
    except Exception:
        return None


def safe_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in (safe_text(v) for v in value) if item]
    text = safe_text(value)
    return [text] if text else []


def to_ascii(value: Any) -> str:
    text = safe_text(value)
    for src, dst in UNICODE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text.encode("ascii", errors="ignore").decode("ascii")


def normalize_text(value: Any) -> str:
    text = to_ascii(value).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9\.\s\-/+]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(value: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?", normalize_text(value))


def singularize(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def meaningful_tokens(value: Any, remove_generic: bool = False) -> List[str]:
    out: List[str] = []
    for token in tokenize(value):
        token = singularize(token)
        if token in STOPWORDS:
            continue
        if remove_generic and token in GENERIC_QUERY_WORDS:
            continue
        out.append(token)
    return out


def build_ngrams(tokens: Sequence[str], n: int) -> Tuple[str, ...]:
    if len(tokens) < n:
        return ()
    return tuple(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def format_price(value: Optional[float]) -> str:
    return f"${value:.2f}" if value is not None else "Not listed in the retrieved dataset."


def format_rating(value: Optional[float]) -> str:
    return f"{value:.1f} / 5.0" if value is not None else "Not listed in the retrieved dataset."


def format_reviews(value: Optional[int]) -> str:
    return str(value) if value is not None else "Not listed in the retrieved dataset."


# ============================================================
# Data model
# ============================================================
@dataclass(frozen=True)
class ProductRecord:
    row_id: int
    raw: Dict[str, Any]
    doc_id: str
    asin: str
    title: str
    main_category: str
    store: str
    price: Optional[float]
    average_rating: Optional[float]
    rating_number: Optional[int]
    categories: Tuple[str, ...]
    features_text: str
    description_text: str
    details_text: str
    embedding_text: str
    full_text: str
    identity_text: str = field(repr=False)
    attribute_text: str = field(repr=False)
    trusted_text: str = field(repr=False)
    normalized_identity_text: str = field(repr=False)
    normalized_attribute_text: str = field(repr=False)
    normalized_search_text: str = field(repr=False)
    title_tokens: Tuple[str, ...] = field(repr=False)
    category_tokens: Tuple[str, ...] = field(repr=False)
    trusted_tokens: Tuple[str, ...] = field(repr=False)
    identity_tokens: Tuple[str, ...] = field(repr=False)
    attribute_tokens: Tuple[str, ...] = field(repr=False)
    search_tokens: Tuple[str, ...] = field(repr=False)
    trusted_bigrams: Tuple[str, ...] = field(repr=False)
    trusted_trigrams: Tuple[str, ...] = field(repr=False)

    @classmethod
    def from_raw(cls, row_id: int, raw: Dict[str, Any]) -> "ProductRecord":
        title = to_ascii(raw.get("title"))
        main_category = to_ascii(raw.get("main_category"))
        store = to_ascii(raw.get("store"))
        asin = to_ascii(raw.get("parent_asin")).upper().strip()
        categories = tuple(to_ascii(x) for x in safe_list(raw.get("categories")))
        features_text = to_ascii(raw.get("features"))
        description_text = to_ascii(raw.get("description"))
        details_text = to_ascii(raw.get("details"))
        embedding_text = to_ascii(raw.get("embedding_text"))
        full_text = to_ascii(raw.get("full_text"))
        category_path = " > ".join(categories)
        identity_text = " ".join(part for part in [title, main_category, category_path] if part)
        attribute_text = " ".join(part for part in [features_text, description_text, details_text, full_text] if part)
        trusted_text = identity_text
        search_text = " ".join(part for part in [identity_text, attribute_text, embedding_text] if part)
        title_tokens = tuple(meaningful_tokens(title, remove_generic=False))
        category_tokens = tuple(meaningful_tokens(main_category + " " + category_path, remove_generic=False))
        trusted_tokens = tuple(meaningful_tokens(trusted_text, remove_generic=False))
        identity_tokens = tuple(meaningful_tokens(identity_text, remove_generic=False))
        attribute_tokens = tuple(meaningful_tokens(attribute_text, remove_generic=False))
        search_tokens = tuple(meaningful_tokens(search_text, remove_generic=False))
        return cls(
            row_id=row_id,
            raw=raw,
            doc_id=safe_text(raw.get("doc_id")) or f"row_{row_id}",
            asin=asin,
            title=title,
            main_category=main_category,
            store=store,
            price=safe_float(raw.get("price")),
            average_rating=safe_float(raw.get("average_rating")),
            rating_number=safe_int(raw.get("rating_number")),
            categories=categories,
            features_text=features_text,
            description_text=description_text,
            details_text=details_text,
            embedding_text=embedding_text,
            full_text=full_text,
            identity_text=identity_text,
            attribute_text=attribute_text,
            trusted_text=trusted_text,
            normalized_identity_text=normalize_text(identity_text),
            normalized_attribute_text=normalize_text(attribute_text),
            normalized_search_text=normalize_text(search_text),
            title_tokens=title_tokens,
            category_tokens=category_tokens,
            trusted_tokens=trusted_tokens,
            identity_tokens=identity_tokens,
            attribute_tokens=attribute_tokens,
            search_tokens=search_tokens,
            trusted_bigrams=build_ngrams(trusted_tokens, 2),
            trusted_trigrams=build_ngrams(trusted_tokens, 3),
        )

    @property
    def category_path_text(self) -> str:
        return " > ".join(self.categories)

    @property
    def key(self) -> str:
        return self.asin or self.doc_id


@dataclass(frozen=True)
class PriceConstraint:
    min_price: Optional[float] = None
    max_price: Optional[float] = None


@dataclass(frozen=True)
class SpecConstraint:
    family: str
    version: float
    mode: str


@dataclass(frozen=True)
class QueryProfile:
    raw_query: str
    normalized_query: str
    mode: str
    sort_by: str
    requested_count: int
    price: PriceConstraint
    compare_targets: Tuple[str, ...]
    asins: Tuple[str, ...]
    query_tokens: Tuple[str, ...]
    query_bigrams: Tuple[str, ...]
    query_trigrams: Tuple[str, ...]
    exact_phrase: str
    spec_constraint: Optional[SpecConstraint]


@dataclass
class Candidate:
    record: ProductRecord
    semantic_score: float = 0.0
    semantic_rrf: float = 0.0
    lexical_score: float = 0.0
    lexical_rrf: float = 0.0
    relevance_score: float = 0.0
    decision_score: float = 0.0
    final_score: float = 0.0
    title_overlap: float = 0.0
    trusted_overlap: float = 0.0
    identity_overlap: float = 0.0
    attribute_overlap: float = 0.0
    exact_phrase_bonus: float = 0.0
    bigram_hits: int = 0
    trigram_hits: int = 0
    spec_bonus: float = 0.0
    llm_label: str = "unjudged"
    llm_confidence: float = 0.0
    llm_reason: str = ""
    matched_queries: List[str] = field(default_factory=list)

    @property
    def fused_rrf(self) -> float:
        return self.semantic_rrf + self.lexical_rrf


# ============================================================
# Repository
# ============================================================
class ProductRepository:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.index = faiss.read_index(str(config.index_path))
        self.records = self._load_records(config.metadata_path)
        if self.index.ntotal != len(self.records):
            raise ValueError(f"Metadata count ({len(self.records)}) does not match FAISS index size ({self.index.ntotal}).")
        self.by_asin: Dict[str, ProductRecord] = {r.asin: r for r in self.records if r.asin}
        self.dimension = self.index.d
        log.info("Loaded %d indexed products.", len(self.records))

    def _load_records(self, metadata_path: Path) -> List[ProductRecord]:
        records: List[ProductRecord] = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                records.append(ProductRecord.from_raw(len(records), json.loads(text)))
        return records

    def get_record(self, index_id: int) -> ProductRecord:
        return self.records[index_id]

    def find_by_asin(self, asin: str) -> Optional[ProductRecord]:
        return self.by_asin.get(asin.upper().strip())


# ============================================================
# Ollama client
# ============================================================
class OllamaClient:
    def __init__(self, config: AppConfig, expected_dimension: int) -> None:
        self.config = config
        self.expected_dimension = expected_dimension
        self.embed_cache: Dict[str, np.ndarray] = {}

    def _post_response(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=self.config.request_timeout, stream=False)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.config.max_retries:
                    raise
                wait_seconds = 2 ** attempt
                log.warning(
                    "Request attempt %d/%d failed for %s: %s. Retrying in %ds.",
                    attempt,
                    self.config.max_retries,
                    url,
                    exc,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
        raise RuntimeError(f"Request failed: {last_error}")

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._post_response(url, payload).json()

    def embed(self, text: str) -> np.ndarray:
        text = safe_text(text)
        if not text:
            raise ValueError("Cannot embed an empty query.")
        if text in self.embed_cache:
            return self.embed_cache[text]
        payload = {"model": self.config.embed_model, "input": [text], "truncate": True}
        data = self._post_json(self.config.embed_url, payload)
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise ValueError(f"Embeddings response missing 'embeddings': {list(data.keys())}")
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != 1:
            raise ValueError(f"Unexpected embedding shape: {array.shape}")
        if array.shape[1] != self.expected_dimension:
            raise ValueError(f"Embedding dim mismatch. Query dim={array.shape[1]}, FAISS dim={self.expected_dimension}.")
        faiss.normalize_L2(array)
        self.embed_cache[text] = array
        return array

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        payload = {
            "model": self.config.chat_model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {"temperature": temperature},
        }
        data = self._post_json(self.config.generate_url, payload)
        content = safe_text(data.get("response"))
        if content:
            return content
        raise ValueError(f"Unexpected generate response: {data}")


# ============================================================
# Query analysis
# ============================================================
class QueryAnalyzer:
    @staticmethod
    def extract_asins(query: str) -> Tuple[str, ...]:
        return tuple(match.upper() for match in ASIN_PATTERN.findall(query.upper()))

    @staticmethod
    def extract_price_constraints(query: str) -> PriceConstraint:
        normalized = normalize_text(query)
        max_price = None
        min_price = None
        for pattern in [
            r"under\s*\$?\s*(\d+(?:\.\d+)?)",
            r"below\s*\$?\s*(\d+(?:\.\d+)?)",
            r"up to\s*\$?\s*(\d+(?:\.\d+)?)",
            r"less than\s*\$?\s*(\d+(?:\.\d+)?)",
            r"at most\s*\$?\s*(\d+(?:\.\d+)?)",
        ]:
            m = re.search(pattern, normalized)
            if m:
                max_price = safe_float(m.group(1))
                break
        for pattern in [
            r"over\s*\$?\s*(\d+(?:\.\d+)?)",
            r"above\s*\$?\s*(\d+(?:\.\d+)?)",
            r"more than\s*\$?\s*(\d+(?:\.\d+)?)",
            r"at least\s*\$?\s*(\d+(?:\.\d+)?)",
        ]:
            m = re.search(pattern, normalized)
            if m:
                min_price = safe_float(m.group(1))
                break
        return PriceConstraint(min_price=min_price, max_price=max_price)

    @staticmethod
    def extract_requested_count(query: str, mode: str, config: AppConfig) -> int:
        m = COUNT_PATTERN.search(query)
        if m:
            return max(1, min(10, int(m.group(1))))
        if mode == MODE_BROWSE:
            return config.default_browse_count
        if mode == MODE_RECOMMEND:
            return config.default_recommend_count
        if mode == MODE_COMPARE:
            return config.default_compare_count
        return 1

    @staticmethod
    def split_compare_targets(query: str) -> Tuple[str, ...]:
        parts = [safe_text(p) for p in COMPARE_PATTERN.split(query) if safe_text(p)]
        return tuple(parts[:2]) if len(parts) >= 2 else ()

    @staticmethod
    def detect_mode(normalized_query: str, asins: Tuple[str, ...], compare_targets: Tuple[str, ...]) -> str:
        if compare_targets or re.search(r"\b(compare|comparison|versus|vs|against)\b", normalized_query):
            return MODE_COMPARE
        if asins:
            return MODE_LOOKUP
        if re.search(r"\b(best|recommend|suggest|ideal|suitable|good for|value|worth)\b", normalized_query):
            return MODE_RECOMMEND
        return MODE_BROWSE

    @staticmethod
    def detect_sort(normalized_query: str, mode: str) -> str:
        if re.search(r"\b(highest rated|top rated|best rated|most stars)\b", normalized_query):
            return SORT_RATING
        if re.search(r"\b(most reviewed|most reviews|most popular|popular)\b", normalized_query):
            return SORT_REVIEWS
        if re.search(r"\b(cheapest|lowest price|least expensive|budget)\b", normalized_query):
            return SORT_PRICE_ASC
        if re.search(r"\b(most expensive|highest price|premium)\b", normalized_query):
            return SORT_PRICE_DESC
        if re.search(r"\b(spec|specs|feature|features|detail|details|performance)\b", normalized_query):
            return SORT_INFO
        if mode == MODE_RECOMMEND:
            return SORT_VALUE
        return SORT_RELEVANCE

    @staticmethod
    def extract_spec_constraint(query: str) -> Optional[SpecConstraint]:
        q = normalize_text(query)
        exact = EXACT_SPEC_PATTERN.search(q)
        minimum = MIN_SPEC_PATTERN.search(q)
        if exact:
            family = exact.group(1).replace("wi-fi", "wifi").replace("usb-c", "usbc")
            return SpecConstraint(family=family, version=float(exact.group(2)), mode=SPEC_EXACT)
        if minimum and minimum.group(2):
            family = safe_text(minimum.group(1)).replace("wi-fi", "wifi").replace("usb-c", "usbc") or ""
            if family:
                return SpecConstraint(family=family, version=float(minimum.group(2)), mode=SPEC_MINIMUM)
        return None

    @classmethod
    def build_profile(cls, query: str, config: AppConfig) -> QueryProfile:
        raw = safe_text(query)
        normalized = normalize_text(raw)
        asins = cls.extract_asins(raw)
        compare_targets = cls.split_compare_targets(raw)
        mode = cls.detect_mode(normalized, asins, compare_targets)
        sort_by = cls.detect_sort(normalized, mode)
        requested_count = cls.extract_requested_count(raw, mode, config)
        price = cls.extract_price_constraints(raw)
        query_tokens = tuple(meaningful_tokens(raw, remove_generic=True))
        exact_phrase = " ".join(query_tokens[:6]).strip()
        return QueryProfile(
            raw_query=raw,
            normalized_query=normalized,
            mode=mode,
            sort_by=sort_by,
            requested_count=requested_count,
            price=price,
            compare_targets=compare_targets,
            asins=asins,
            query_tokens=query_tokens,
            query_bigrams=build_ngrams(query_tokens, 2),
            query_trigrams=build_ngrams(query_tokens, 3),
            exact_phrase=exact_phrase,
            spec_constraint=cls.extract_spec_constraint(raw),
        )


# ============================================================
# Retrieval
# ============================================================
class HybridRetriever:
    def __init__(self, repository: ProductRepository, ollama: OllamaClient, config: AppConfig) -> None:
        self.repository = repository
        self.ollama = ollama
        self.config = config

    def build_query_expansions(self, profile: QueryProfile) -> List[str]:
        expansions = [profile.raw_query]
        if profile.compare_targets:
            expansions.extend(profile.compare_targets)
        if profile.query_trigrams:
            expansions.extend(profile.query_trigrams[:3])
        if profile.query_bigrams:
            expansions.extend(profile.query_bigrams[:3])
        if profile.query_tokens:
            expansions.append(" ".join(profile.query_tokens[:6]))
        budgetless = re.sub(
            r"\b(under|below|up to|less than|at most|over|above|more than|at least)\b\s*\$?\s*\d+(?:\.\d+)?",
            " ",
            profile.normalized_query,
        )
        budgetless = re.sub(r"\s+", " ", budgetless).strip()
        if budgetless and budgetless != profile.normalized_query:
            expansions.append(budgetless)
        unique: List[str] = []
        seen: Set[str] = set()
        for e in expansions:
            text = safe_text(e)
            if not text:
                continue
            key = text.lower()
            if key not in seen:
                seen.add(key)
                unique.append(text)
        return unique[:8]

    def semantic_search(self, text: str, limit: int) -> List[Tuple[ProductRecord, float]]:
        vec = self.ollama.embed(text)
        scores, indices = self.repository.index.search(vec, limit)
        out: List[Tuple[ProductRecord, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.repository.records):
                continue
            if float(score) < self.config.semantic_min_score:
                continue
            out.append((self.repository.get_record(int(idx)), float(score)))
        return out

    def lexical_search(self, profile: QueryProfile) -> List[Tuple[ProductRecord, float]]:
        query_tokens = set(profile.query_tokens)
        query_bigrams = set(profile.query_bigrams)
        query_trigrams = set(profile.query_trigrams)
        if not query_tokens and not query_bigrams and not query_trigrams:
            return []
        scored: List[Tuple[float, ProductRecord]] = []
        for record in self.repository.records:
            trusted = set(record.trusted_tokens)
            identity = set(record.identity_tokens)
            attribute = set(record.attribute_tokens)
            score = 0.0
            score += 4.0 * len(query_trigrams & set(record.trusted_trigrams))
            score += 2.6 * len(query_bigrams & set(record.trusted_bigrams))
            score += 1.9 * len(query_tokens & trusted)
            score += 1.4 * len(query_tokens & identity)
            score += 0.12 * len(query_tokens & attribute)
            if profile.exact_phrase and profile.exact_phrase in record.trusted_text:
                score += 5.0
            elif profile.exact_phrase and profile.exact_phrase in record.normalized_identity_text:
                score += 3.0
            elif profile.exact_phrase and profile.exact_phrase in record.normalized_attribute_text:
                score += 0.75
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [(r, s) for s, r in scored[: self.config.lexical_top_k]]

    def retrieve(self, profile: QueryProfile) -> List[Candidate]:
        merged: Dict[str, Candidate] = {}
        for expansion in self.build_query_expansions(profile):
            for rank, (record, score) in enumerate(self.semantic_search(expansion, self.config.semantic_top_k), start=1):
                key = record.key
                candidate = merged.get(key)
                if candidate is None:
                    candidate = Candidate(record=record)
                    merged[key] = candidate
                candidate.semantic_score = max(candidate.semantic_score, score)
                candidate.semantic_rrf += 1.0 / (50 + rank)
                candidate.matched_queries.append(expansion)
        for rank, (record, score) in enumerate(self.lexical_search(profile), start=1):
            key = record.key
            candidate = merged.get(key)
            if candidate is None:
                candidate = Candidate(record=record)
                merged[key] = candidate
            candidate.lexical_score = max(candidate.lexical_score, score)
            candidate.lexical_rrf += 1.0 / (50 + rank)
        candidates = list(merged.values())
        candidates.sort(key=lambda c: (-(c.semantic_rrf + c.lexical_rrf), -c.semantic_score, -c.lexical_score))
        return candidates[: self.config.candidate_pool]


# ============================================================
# Reranking and selection
# ============================================================
class Reranker:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @staticmethod
    def overlap_ratio(query_tokens: Sequence[str], target_tokens: Sequence[str]) -> float:
        q = set(query_tokens)
        if not q:
            return 0.0
        return len(q & set(target_tokens)) / len(q)

    @staticmethod
    def credibility(record: ProductRecord) -> float:
        rating = record.average_rating or 0.0
        reviews = record.rating_number or 0
        if rating <= 0 or reviews <= 0:
            return 0.0
        return rating * math.log1p(reviews)

    @staticmethod
    def information(record: ProductRecord) -> float:
        score = 0.0
        if record.features_text:
            score += 1.2
        if record.details_text:
            score += 1.2
        if record.description_text:
            score += 0.8
        if record.full_text:
            score += 0.6
        if record.average_rating is not None:
            score += 0.4
        if record.rating_number is not None:
            score += 0.4
        if record.price is not None:
            score += 0.4
        return score

    @staticmethod
    def value_signal(record: ProductRecord) -> float:
        cred = Reranker.credibility(record)
        info = Reranker.information(record)
        if record.price is None or record.price <= 0:
            return 0.35 * min(cred / 20.0, 1.0) + 0.25 * min(info / 5.0, 1.0)
        return (0.75 * cred + 0.35 * info) / math.sqrt(max(record.price, 1.0))

    @staticmethod
    def spec_bonus(profile: QueryProfile, record: ProductRecord) -> float:
        if not profile.spec_constraint:
            return 0.0
        family = re.escape(profile.spec_constraint.family)
        pattern = re.compile(rf"\b{family}\s*([0-9]+(?:\.[0-9]+)?)\b", flags=re.IGNORECASE)
        versions: List[float] = []
        for text in [record.title, record.features_text, record.details_text, record.description_text, record.full_text]:
            for m in pattern.finditer(to_ascii(text)):
                try:
                    versions.append(float(m.group(1)))
                except Exception:
                    continue
        if not versions:
            return 0.0
        requested = profile.spec_constraint.version
        if profile.spec_constraint.mode == SPEC_EXACT:
            if any(abs(v - requested) < 1e-6 for v in versions):
                return 3.0
            if any(v > requested for v in versions):
                return 0.25
            return -0.5
        if any(v >= requested for v in versions):
            closest = min(v for v in versions if v >= requested)
            return 2.5 - 0.2 * max(closest - requested, 0)
        return -1.0

    def apply_constraints(self, candidates: List[Candidate], profile: QueryProfile) -> List[Candidate]:
        out: List[Candidate] = []
        for candidate in candidates:
            record = candidate.record
            if profile.price.max_price is not None and record.price is not None and record.price > profile.price.max_price:
                continue
            if profile.price.min_price is not None and record.price is not None and record.price < profile.price.min_price:
                continue
            out.append(candidate)
        return out

    def score_relevance(self, candidates: List[Candidate], profile: QueryProfile) -> None:
        for candidate in candidates:
            record = candidate.record
            candidate.title_overlap = self.overlap_ratio(profile.query_tokens, record.title_tokens)
            candidate.trusted_overlap = self.overlap_ratio(profile.query_tokens, record.trusted_tokens)
            candidate.identity_overlap = self.overlap_ratio(profile.query_tokens, record.identity_tokens)
            candidate.attribute_overlap = self.overlap_ratio(profile.query_tokens, record.attribute_tokens)
            candidate.bigram_hits = len(set(profile.query_bigrams) & set(record.trusted_bigrams))
            candidate.trigram_hits = len(set(profile.query_trigrams) & set(record.trusted_trigrams))
            candidate.exact_phrase_bonus = 0.0
            if profile.exact_phrase:
                if profile.exact_phrase in record.trusted_text:
                    candidate.exact_phrase_bonus = 3.0
                elif profile.exact_phrase in record.normalized_identity_text:
                    candidate.exact_phrase_bonus = 2.0
                elif profile.exact_phrase in record.normalized_attribute_text:
                    candidate.exact_phrase_bonus = 0.5
            candidate.spec_bonus = self.spec_bonus(profile, record)
            candidate.relevance_score = (
                2.5 * candidate.semantic_score
                + 1.15 * candidate.fused_rrf
                + 2.5 * candidate.identity_overlap
                + 1.9 * candidate.trusted_overlap
                + 1.2 * candidate.title_overlap
                + 0.35 * candidate.attribute_overlap
                + 0.8 * candidate.bigram_hits
                + 1.1 * candidate.trigram_hits
                + candidate.exact_phrase_bonus
                + candidate.spec_bonus
            )

    def score_decision(self, candidates: List[Candidate], profile: QueryProfile) -> None:
        for candidate in candidates:
            record = candidate.record
            rating = record.average_rating or 0.0
            reviews = record.rating_number or 0
            credibility = self.credibility(record)
            information = self.information(record)
            value_signal = self.value_signal(record)
            price = record.price

            if profile.sort_by == SORT_RATING:
                candidate.decision_score = 1.8 * rating + 0.35 * math.log1p(reviews)
            elif profile.sort_by == SORT_REVIEWS:
                candidate.decision_score = 1.2 * math.log1p(reviews) + 0.25 * rating
            elif profile.sort_by == SORT_PRICE_ASC:
                candidate.decision_score = -0.025 * price if price is not None else 0.0
            elif profile.sort_by == SORT_PRICE_DESC:
                candidate.decision_score = 0.015 * price if price is not None else 0.0
            elif profile.sort_by == SORT_INFO:
                candidate.decision_score = 0.95 * information + 0.20 * rating
            elif profile.sort_by == SORT_VALUE:
                candidate.decision_score = 1.25 * value_signal + 0.15 * math.log1p(reviews)
            else:
                candidate.decision_score = (
                    0.70 * value_signal
                    + 0.40 * min(credibility / 20.0, 1.0)
                    + 0.30 * min(information / 5.0, 1.0)
                )
            candidate.final_score = 1.8 * candidate.relevance_score + candidate.decision_score

    def rank(self, candidates: List[Candidate], profile: QueryProfile) -> List[Candidate]:
        filtered = self.apply_constraints(candidates, profile)
        self.score_relevance(filtered, profile)
        self.score_decision(filtered, profile)
        filtered.sort(key=lambda c: (-c.final_score, -c.relevance_score, -c.semantic_score, -c.fused_rrf))
        return filtered

    @staticmethod
    def signature(record: ProductRecord) -> Set[str]:
        return {t for t in record.title_tokens + record.category_tokens if t not in STOPWORDS}

    @staticmethod
    def jaccard(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / max(len(a | b), 1)

    def select_diverse(self, ranked: List[Candidate], limit: int) -> List[Candidate]:
        if len(ranked) <= limit:
            return ranked
        selected: List[Candidate] = []
        signatures: List[Set[str]] = []
        for cand in ranked:
            sig = self.signature(cand.record)
            if not selected:
                selected.append(cand)
                signatures.append(sig)
                continue
            max_sim = max((self.jaccard(sig, prev) for prev in signatures), default=0.0)
            if max_sim < 0.72 or len(selected) + 1 > limit:
                selected.append(cand)
                signatures.append(sig)
            if len(selected) >= limit:
                break
        if len(selected) < limit:
            for cand in ranked:
                if cand not in selected:
                    selected.append(cand)
                if len(selected) >= limit:
                    break
        return selected


# ============================================================
# Evidence formatting and prompting
# ============================================================
class EvidenceBuilder:
    LABEL_LINE_PREFIXES = (
        "title:",
        "main category:",
        "brand/store:",
        "store:",
        "categories:",
        "price:",
        "average rating:",
        "rating number:",
        "number of reviews:",
        "asin:",
        "parent asin:",
        "embedding text:",
        "full text:",
    )

    CATEGORY_LABELS = {
        "electronics",
        "computers",
        "computers & accessories",
        "laptop accessories",
        "home audio & theater",
        "television & video",
        "portable audio & video",
        "accessories",
        "all electronics",
        "industrial & scientific",
        "camera & photo",
        "security & surveillance",
        "security cameras",
    }

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _extract_category_lines_from_full_text(self, record: ProductRecord) -> List[str]:
        if record.categories:
            return list(record.categories)
        text = record.full_text or ""
        if not text:
            return []
        lines = [to_ascii(line).strip() for line in text.splitlines() if line.strip()]
        out: List[str] = []
        seen: Set[str] = set()
        in_categories = False
        for line in lines:
            norm = normalize_text(line)
            if norm == "categories":
                in_categories = True
                continue
            if in_categories:
                if norm.endswith(":") or norm in {"title", "main category", "brand store", "price", "average rating", "rating number", "parent asin"}:
                    break
                if norm and (norm in self.CATEGORY_LABELS or len(out) < 4):
                    if norm not in seen:
                        seen.add(norm)
                        out.append(line)
                if len(out) >= 5:
                    break
        return out

    def extract_feature_lines(self, record: ProductRecord) -> List[str]:
        primary_text = "\n".join(part for part in [record.features_text, record.details_text, record.description_text] if part)
        fallback_text = record.full_text or ""
        text = primary_text if primary_text.strip() else fallback_text
        if not text:
            return []
        raw_pieces = [part.strip(" -*\t") for part in re.split(r"[\n;|]+", text) if part.strip()]
        clean: List[str] = []
        seen: Set[str] = set()
        for piece in raw_pieces:
            ascii_piece = to_ascii(piece).strip()
            norm = normalize_text(ascii_piece)
            if not norm or norm in {"none", "not available", "n a"}:
                continue
            if any(norm.startswith(prefix) for prefix in self.LABEL_LINE_PREFIXES):
                continue
            if norm in self.CATEGORY_LABELS:
                continue
            if len(norm) < 3:
                continue
            # Skip extremely generic fragments
            if norm in {"electronics", "accessories", "computer", "portable audio and video"}:
                continue
            if norm not in seen:
                seen.add(norm)
                clean.append(ascii_piece)
            if len(clean) >= self.config.max_feature_lines:
                break
        return clean

    def product_summary(self, record: ProductRecord) -> Dict[str, Any]:
        category_lines = list(record.categories) if record.categories else self._extract_category_lines_from_full_text(record)
        categories_text = " > ".join(category_lines) if category_lines else "Not listed in the retrieved dataset."
        return {
            "title": record.title or "Not listed in the retrieved dataset.",
            "asin": record.asin or "Not listed in the retrieved dataset.",
            "brand_store": record.store or "Not listed in the retrieved dataset.",
            "main_category": record.main_category or "Not listed in the retrieved dataset.",
            "categories": categories_text,
            "price": format_price(record.price),
            "average_rating": format_rating(record.average_rating),
            "number_of_reviews": format_reviews(record.rating_number),
            "feature_lines": self.extract_feature_lines(record),
        }

    def render_product_for_llm(self, index: int, candidate: Candidate) -> str:
        s = self.product_summary(candidate.record)
        feature_block = "\n".join(f"- {line}" for line in s["feature_lines"]) if s["feature_lines"] else "- Not listed in the retrieved dataset."
        return (
            f"Product {index}\n"
            f"Title: {s['title']}\n"
            f"ASIN: {s['asin']}\n"
            f"Brand/Store: {s['brand_store']}\n"
            f"Main category: {s['main_category']}\n"
            f"Categories: {s['categories']}\n"
            f"Price: {s['price']}\n"
            f"Average rating: {s['average_rating']}\n"
            f"Number of reviews: {s['number_of_reviews']}\n"
            f"Feature lines:\n{feature_block}\n"
        )

    def compare_prompt(self, query: str, candidates: List[Candidate]) -> str:
        blocks = "\n\n".join(self.render_product_for_llm(i + 1, c) for i, c in enumerate(candidates[:2]))
        return (
            f"User query: {query}\n\n"
            f"Retrieved grounded products:\n\n{blocks}\n\n"
            "Compare only these two products. Use only the fields shown above. "
            "Do not invent any information. If a field is missing, say 'Not listed in the retrieved dataset.' "
            "State trade-offs clearly and end with a cautious conclusion about which one better matches the user's request."
        )

    def recommend_prompt(self, query: str, candidates: List[Candidate], count: int) -> str:
        blocks = "\n\n".join(self.render_product_for_llm(i + 1, c) for i, c in enumerate(candidates))
        return (
            f"User query: {query}\n"
            f"Return exactly {count} recommendation(s).\n\n"
            f"Retrieved grounded products:\n\n{blocks}\n\n"
            "Choose the best recommendations for the user's need using only the provided products. "
            "Prioritize relevance first, then the user's requested preference such as value, price, rating, reviews, or specs. "
            "If price or any other field is missing, treat it as unknown instead of assuming it is bad. "
            "For each chosen product, explain why it fits using only evidence above. "
            "Do not invent any specifications or claims. If evidence is limited, say so clearly."
        )

    def browse_prompt(self, query: str, candidates: List[Candidate], count: int) -> str:
        blocks = "\n\n".join(self.render_product_for_llm(i + 1, c) for i, c in enumerate(candidates))
        return (
            f"User query: {query}\n"
            f"Return exactly {count} relevant product(s).\n\n"
            f"Retrieved grounded products:\n\n{blocks}\n\n"
            "Present the most relevant products for the query using only the evidence above. "
            "Do not invent missing information. If support is limited, say so. Keep the answer clear and structured."
        )

    def lookup_text(self, record: ProductRecord) -> str:
        s = self.product_summary(record)
        lines = [
            f"Title: {s['title']}",
            f"ASIN: {s['asin']}",
            f"Brand/Store: {s['brand_store']}",
            f"Main category: {s['main_category']}",
            f"Categories: {s['categories']}",
            f"Price: {s['price']}",
            f"Average rating: {s['average_rating']}",
            f"Number of reviews: {s['number_of_reviews']}",
        ]
        if s["feature_lines"]:
            lines.append("Features:")
            lines.extend(f"- {line}" for line in s["feature_lines"])
        else:
            lines.append("Features: Not listed in the retrieved dataset.")
        return "\n".join(lines)


# ============================================================
# LLM candidate judge and answerer
# ============================================================
class LLMCandidateJudge:
    def __init__(self, ollama: OllamaClient, evidence: EvidenceBuilder, config: AppConfig) -> None:
        self.ollama = ollama
        self.evidence = evidence
        self.config = config
        self.system_prompt = (
            "You are a grounded retrieval judge in a product RAG pipeline. "
            "You must decide whether a retrieved candidate is itself a direct answer to the user query, "
            "or only a related item such as an accessory, a weak semantic match, or something unrelated. "
            "Use only the provided product evidence. Do not use outside knowledge. "
            "Return compact JSON only with keys: label, confidence, reason. "
            "Allowed labels: direct_match, related_accessory, weak_match, unrelated. "
            "Choose direct_match only when the product itself directly satisfies the requested item, not merely something related to it."
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = safe_text(text)
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}
        return {}

    def _heuristic_fallback(self, query: str, candidate: Candidate) -> Tuple[str, float, str]:
        q = set(meaningful_tokens(query, remove_generic=True))
        title = set(candidate.record.title_tokens)
        trusted = set(candidate.record.trusted_tokens)
        title_overlap = len(q & title)
        trusted_overlap = len(q & trusted)
        if title_overlap >= 2 and trusted_overlap >= 2:
            return (LABEL_WEAK, 0.45, "fallback partial identity support")
        if title_overlap >= 1 or trusted_overlap >= 1:
            return (LABEL_WEAK, 0.30, "fallback weak identity support")
        return (LABEL_UNRELATED, 0.15, "fallback no direct identity support")

    def judge(self, query: str, candidate: Candidate) -> Candidate:
        product_block = self.evidence.render_product_for_llm(1, candidate)
        prompt = (
            f"User query: {query}\n\n"
            f"Candidate product:\n\n{product_block}\n\n"
            "Decide whether this candidate itself is a direct answer to the user query. "
            "Do not answer the final user question. Return JSON only."
        )
        try:
            raw = self.ollama.chat(self.system_prompt, prompt, temperature=0.0)
            data = self._extract_json(raw)
            label = safe_text(data.get("label")).lower()
            if label not in VALID_LABELS:
                raise ValueError("invalid label")
            confidence = safe_float(data.get("confidence"))
            confidence = max(0.0, min(confidence if confidence is not None else 0.5, 1.0))
            reason = safe_text(data.get("reason")) or "No reason provided."
            candidate.llm_label = label
            candidate.llm_confidence = confidence
            candidate.llm_reason = reason
            return candidate
        except Exception:
            label, confidence, reason = self._heuristic_fallback(query, candidate)
            candidate.llm_label = label
            candidate.llm_confidence = confidence
            candidate.llm_reason = reason
            return candidate

    def filter_shortlist(self, query: str, candidates: List[Candidate], profile: QueryProfile) -> List[Candidate]:
        if profile.mode == MODE_LOOKUP:
            return candidates
        judged = [self.judge(query, candidate) for candidate in candidates]
        direct = [c for c in judged if c.llm_label == LABEL_DIRECT and c.llm_confidence >= 0.70]
        weak = [c for c in judged if c.llm_label == LABEL_WEAK and c.llm_confidence >= 0.45]
        direct.sort(key=lambda c: (-c.llm_confidence, -c.final_score, -c.relevance_score))
        weak.sort(key=lambda c: (-c.llm_confidence, -c.final_score, -c.relevance_score))
        if profile.mode == MODE_COMPARE:
            return candidates
        target_count = self.config.default_browse_count if profile.mode == MODE_BROWSE else max(profile.requested_count, self.config.default_recommend_count)
        if direct:
            return direct[:target_count]
        if weak:
            return weak[:min(target_count, 1)]
        return []


class GroundedAnswerer:
    def __init__(self, ollama: OllamaClient, config: AppConfig) -> None:
        self.ollama = ollama
        self.config = config
        self.system_prompt = (
            "You are a grounded product RAG assistant. "
            "You must answer strictly from the retrieved product evidence provided by the user prompt. "
            "Never invent or infer missing product facts. "
            "If a value is missing, say 'Not listed in the retrieved dataset.' "
            "Do not use outside knowledge. "
            "Be careful with version/spec requests: if the user asks for an exact version, prefer exact evidence instead of assuming newer is better. "
            "Keep answers clear, honest, and evidence-based."
        )

    def answer(self, prompt: str) -> str:
        return self.ollama.chat(self.system_prompt, prompt, temperature=0.0)


# ============================================================
# Orchestrator
# ============================================================
class AmazonElectronicsRealRAG:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.repository = ProductRepository(config)
        self.ollama = OllamaClient(config, expected_dimension=self.repository.dimension)
        self.analyzer = QueryAnalyzer()
        self.retriever = HybridRetriever(self.repository, self.ollama, config)
        self.reranker = Reranker(config)
        self.evidence = EvidenceBuilder(config)
        self.candidate_judge = LLMCandidateJudge(self.ollama, self.evidence, config)
        self.answerer = GroundedAnswerer(self.ollama, config)

    def resolve_asin_lookup(self, profile: QueryProfile) -> Optional[ProductRecord]:
        for asin in profile.asins:
            record = self.repository.find_by_asin(asin)
            if record:
                return record
        return None

    def resolve_compare_targets(self, profile: QueryProfile, ranked: List[Candidate]) -> List[Candidate]:
        if not profile.compare_targets:
            return ranked[:2]
        chosen: List[Candidate] = []
        used: Set[str] = set()
        direct_records: List[ProductRecord] = []
        for asin in profile.asins:
            record = self.repository.find_by_asin(asin)
            if record and record.key not in used:
                direct_records.append(record)
                used.add(record.key)
        for record in direct_records:
            matched_existing = next((c for c in ranked if c.record.key == record.key), None)
            chosen.append(matched_existing if matched_existing is not None else Candidate(record=record))
        if len(chosen) >= 2:
            return chosen[:2]
        pool = ranked[: max(8, self.config.llm_shortlist_k)]
        for target in profile.compare_targets[:2]:
            target_asins = tuple(match.upper() for match in ASIN_PATTERN.findall(target.upper()))
            target_tokens = tuple(meaningful_tokens(target, remove_generic=True))
            best: Optional[Candidate] = None
            best_score = -1e9
            for candidate in pool:
                if candidate.record.key in used:
                    continue
                score = 0.0
                if target_asins and candidate.record.asin in target_asins:
                    score += 100.0
                score += 3.0 * Reranker.overlap_ratio(target_tokens, candidate.record.title_tokens)
                score += 2.0 * Reranker.overlap_ratio(target_tokens, candidate.record.trusted_tokens)
                normalized_target = normalize_text(target)
                if normalized_target and normalized_target in candidate.record.trusted_text:
                    score += 4.0
                if score > best_score:
                    best_score = score
                    best = candidate
            if best is not None:
                chosen.append(best)
                used.add(best.record.key)
            if len(chosen) >= 2:
                return chosen[:2]
        for candidate in ranked:
            if candidate.record.key not in used:
                chosen.append(candidate)
                used.add(candidate.record.key)
            if len(chosen) >= 2:
                break
        return chosen[:2]

    def shortlist_for_llm(self, ranked: List[Candidate], profile: QueryProfile, query: str) -> List[Candidate]:
        if profile.mode == MODE_COMPARE:
            shortlist = self.resolve_compare_targets(profile, ranked)
            return shortlist[: self.config.default_compare_count]
        if profile.mode == MODE_RECOMMEND:
            diverse = self.reranker.select_diverse(ranked[: self.config.llm_shortlist_k * 2], self.config.llm_shortlist_k)
            shortlist = diverse[: self.config.llm_shortlist_k]
            return self.candidate_judge.filter_shortlist(query, shortlist, profile)
        if profile.mode == MODE_BROWSE:
            diverse = self.reranker.select_diverse(ranked[: self.config.llm_shortlist_k * 2], self.config.default_browse_count + 3)
            shortlist = diverse[: self.config.llm_shortlist_k]
            return self.candidate_judge.filter_shortlist(query, shortlist, profile)
        return ranked[: self.config.llm_shortlist_k]

    def run(self, query: str) -> Generator[str, None, None]:
        query = safe_text(query)
        if not query:
            yield "Please enter a question."
            return
        profile = self.analyzer.build_profile(query, self.config)
        if profile.mode == MODE_LOOKUP:
            record = self.resolve_asin_lookup(profile)
            if record is None:
                yield "I could not find that ASIN in the retrieved dataset."
                return
            prompt = (
                f"User query: {query}\n\n"
                f"Retrieved grounded product:\n\n{self.evidence.lookup_text(record)}\n\n"
                "Give a concise grounded summary of this product using only the retrieved evidence above. "
                "Do not invent missing information."
            )
            answer = self.answerer.answer(prompt)
            yield f"{self.evidence.lookup_text(record)}\n\nGrounded summary:\n{answer}"
            return
        retrieved = self.retriever.retrieve(profile)
        ranked = self.reranker.rank(retrieved, profile)
        if not ranked:
            yield (
                "I couldn't find any grounded products matching your query.\n\n"
                "Try:\n"
                "- using a clearer product or category phrase\n"
                "- removing or changing the price limit\n"
                "- giving an exact product name\n"
                "- providing an ASIN if you have one"
            )
            return
        shortlist = self.shortlist_for_llm(ranked, profile, query)
        if not shortlist:
            yield "I retrieved candidates, but I could not build a confident grounded shortlist for this query."
            return
        if profile.mode == MODE_COMPARE:
            prompt = self.evidence.compare_prompt(query, shortlist[:2])
            yield self.answerer.answer(prompt)
            return
        if profile.mode == MODE_RECOMMEND:
            prompt = self.evidence.recommend_prompt(query, shortlist, profile.requested_count)
            yield self.answerer.answer(prompt)
            return
        prompt = self.evidence.browse_prompt(query, shortlist, profile.requested_count)
        yield self.answerer.answer(prompt)


# ============================================================
# Gradio UI
# ============================================================
def build_ui(app: AmazonElectronicsRealRAG) -> gr.Blocks:
    with gr.Blocks(title="Intelligent RAG Chatbot For Electronics") as demo:
        gr.Markdown(
            "## Intelligent RAG Chatbot For Electronics\n"
            "Find the right electronics product faster with evidence-based answers built only from retrieved product data."
        )
        chatbot = gr.Chatbot(height=520)
        state = gr.State([])
        with gr.Row():
            message_box = gr.Textbox(
                placeholder=(
                    "Examples: 'Tell me about ASIN B096WG4SML' | 'Compare product A vs product B' | "
                    "'Recommend the best HDMI cable for value' | 'Show 3 audio cables'"
                ),
                lines=2,
                scale=8,
                show_label=False,
            )
            send_button = gr.Button("Send", variant="primary", scale=1)
        clear_button = gr.Button("Clear", variant="secondary")

        def user_submit(user_message: str, history: list) -> tuple[str, list]:
            cleaned = safe_text(user_message)
            if not cleaned:
                return "", history
            return "", history + [{"role": "user", "content": cleaned}]

        def bot_respond(history: list):
            user_message = next(msg["content"] for msg in reversed(history) if msg["role"] == "user")
            updated_history = history + [{"role": "assistant", "content": ""}]
            try:
                for partial in app.run(user_message):
                    updated_history[-1] = {"role": "assistant", "content": partial}
                    yield updated_history
            except requests.ConnectionError:
                updated_history[-1] = {
                    "role": "assistant",
                    "content": (
                        "**Error:** Cannot reach Ollama. Run `ollama serve` and make sure both models are available:\n"
                        f"- Embeddings: `{CONFIG.embed_model}`\n"
                        f"- Chat: `{CONFIG.chat_model}`"
                    ),
                }
                yield updated_history
            except Exception as exc:
                log.exception("Unhandled error while processing user message")
                updated_history[-1] = {"role": "assistant", "content": f"**Unexpected error:** {exc}"}
                yield updated_history

        message_box.submit(user_submit, [message_box, state], [message_box, state]).then(bot_respond, state, chatbot).then(
            lambda history: history, chatbot, state
        )
        send_button.click(user_submit, [message_box, state], [message_box, state]).then(bot_respond, state, chatbot).then(
            lambda history: history, chatbot, state
        )
        clear_button.click(lambda: ([], []), None, [chatbot, state])
        gr.Markdown(
            "**Note:** Every answer is grounded in retrieved product records only — no hallucinations, no external knowledge, and no guessed missing details."
        )
    return demo


# ============================================================
# Entry point
# ============================================================
def main() -> None:
    app = AmazonElectronicsRealRAG(CONFIG)
    demo = build_ui(app)
    demo.queue()
    demo.launch(share=True)


if __name__ == "__main__":
    main()
