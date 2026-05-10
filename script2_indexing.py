"""
Script 2 — Embedding & FAISS Indexing
=======================================
Reads the cleaned JSONL from Script 1, embeds every record with Ollama,
and writes a FAISS index + aligned metadata JSONL to disk.

Key improvements over original:
  - Retry logic with exponential back-off on every Ollama call
  - Checkpointing: resumes from the last successful batch on failure
  - IVFFlat index (with flat fallback for small datasets) for scalability
  - Token-aware truncation note baked in (bge-base = 512 tokens)
  - Progress reporting via tqdm (falls back gracefully if not installed)
  - Config-only top section; no magic numbers scattered through the code
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import requests

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
INPUT_RECORDS_PATH  = Path(r"C:\Users\pola\Desktop\RAG project\records_sample.jsonl")
INDEX_OUTPUT_PATH   = Path(r"C:\Users\pola\Desktop\RAG project\output_data\electronics_index1.faiss")
METADATA_OUTPUT_PATH= Path(r"C:\Users\pola\Desktop\RAG project\output_data\metadata_map1.jsonl")
CHECKPOINT_PATH     = Path(r"C:\Users\pola\Desktop\RAG project\output_data\embed_checkpoint.npy")

OLLAMA_EMBED_URL   = "http://localhost:11434/api/embed"
EMBEDDING_MODEL    = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"

BATCH_SIZE         = 32
REQUEST_TIMEOUT    = 300       # seconds per batch call
MAX_RETRIES        = 5
RETRY_BASE_DELAY   = 2.0       # seconds; doubles each retry

# bge-base-en-v1.5 hard limit is 512 tokens.
# At ~4 chars/token, 1 800 chars gives comfortable headroom.
MAX_EMBED_CHARS    = 1_800

# IVFFlat: use when ntotal >= this threshold; otherwise use flat index.
IVF_THRESHOLD      = 1_000
IVF_NLIST          = 100       # number of Voronoi cells


# ──────────────────────────────────────────────
# JSONL I/O
# ──────────────────────────────────────────────
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Saved %d metadata records → %s", len(records), path)


# ──────────────────────────────────────────────
# Text prep
# ──────────────────────────────────────────────
def prepare_text(text: str) -> str:
    """Truncate to stay within the model's token budget."""
    text = str(text).strip()
    return text[:MAX_EMBED_CHARS].rstrip()


def filter_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop records whose embedding_text is too short to be useful."""
    valid = [r for r in records if len(str(r.get("embedding_text", "")).strip()) >= 20]
    log.info("Filter: %d / %d records kept", len(valid), len(records))
    return valid


# ──────────────────────────────────────────────
# Ollama embedding  (with retry + back-off)
# ──────────────────────────────────────────────
def embed_batch(texts: List[str]) -> np.ndarray:
    """
    Call Ollama's /api/embed endpoint with retry and exponential back-off.
    Returns an (N, D) float32 array, L2-normalised (cosine similarity ready).
    """
    payload = {
        "model":    EMBEDDING_MODEL,
        "input":    texts,
        "truncate": True,
    }

    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            if "embeddings" not in data:
                raise ValueError(f"Missing 'embeddings' in response: {list(data.keys())}")

            vecs = np.asarray(data["embeddings"], dtype=np.float32)
            if vecs.ndim != 2 or vecs.shape[0] != len(texts):
                raise ValueError(f"Shape mismatch: expected ({len(texts)}, D), got {vecs.shape}")

            faiss.normalize_L2(vecs)
            return vecs

        except Exception as exc:
            last_exc = exc
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log.warning("Embed attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt, MAX_RETRIES, exc, delay)
            time.sleep(delay)

    raise RuntimeError(f"All {MAX_RETRIES} embed attempts failed. Last error: {last_exc}")


def embed_all(records: List[Dict[str, Any]]) -> np.ndarray:
    """
    Embed every record in batches, with optional checkpoint resume.
    Saves a .npy checkpoint after each batch so a crash is recoverable.
    """
    texts  = [prepare_text(r["embedding_text"]) for r in records]
    n      = len(texts)
    n_batches = math.ceil(n / BATCH_SIZE)

    # Resume from checkpoint if it exists
    start_batch = 0
    all_vecs: List[np.ndarray] = []

    if CHECKPOINT_PATH.exists():
        try:
            saved = np.load(str(CHECKPOINT_PATH))
            n_done = saved.shape[0]
            start_batch = n_done // BATCH_SIZE
            all_vecs = [saved]
            log.info("Resuming from checkpoint: %d vectors already embedded", n_done)
        except Exception as e:
            log.warning("Could not load checkpoint (%s), starting fresh.", e)

    iterator = range(start_batch, n_batches)
    if HAS_TQDM:
        iterator = tqdm(iterator, desc="Embedding", unit="batch", initial=start_batch, total=n_batches)

    for batch_idx in iterator:
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, n)
        batch_texts = texts[start:end]

        log.info("Batch %d/%d  (records %d–%d)", batch_idx + 1, n_batches, start, end - 1)
        vecs = embed_batch(batch_texts)
        all_vecs.append(vecs)

        # Save checkpoint after every batch
        combined = np.vstack(all_vecs)
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(CHECKPOINT_PATH), combined)

    result = np.vstack(all_vecs)
    log.info("Embedding complete: shape %s", result.shape)

    # Clean up checkpoint now that we succeeded
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    return result


# ──────────────────────────────────────────────
# FAISS index builder
# ──────────────────────────────────────────────
def build_index(vecs: np.ndarray) -> faiss.Index:
    """
    Build an appropriate FAISS index for the dataset size.
    - Small datasets  (< IVF_THRESHOLD): IndexFlatIP — exact, no training needed.
    - Larger datasets: IndexIVFFlat — approximate, much faster at search time.
    Both use inner-product (cosine, since vectors are L2-normalised).
    """
    n, dim = vecs.shape
    log.info("Building FAISS index  (n=%d, dim=%d)", n, dim)

    if n < IVF_THRESHOLD:
        log.info("Using IndexFlatIP (exact search) for small dataset")
        index = faiss.IndexFlatIP(dim)
    else:
        nlist = min(IVF_NLIST, n // 10)   # cells must be < n
        log.info("Using IndexIVFFlat  (nlist=%d) for larger dataset", nlist)
        quantiser = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vecs)

    index.add(vecs)
    log.info("Index built: %d vectors", index.ntotal)
    return index


# ──────────────────────────────────────────────
# Metadata builder
# ──────────────────────────────────────────────
def build_metadata(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build aligned metadata list (index i corresponds to FAISS vector i).
    Stores all fields needed for filtering, display, and LLM prompting.
    """
    return [
        {
            "faiss_index":    i,
            "doc_id":         r.get("doc_id"),
            "parent_asin":    r.get("parent_asin"),
            "title":          r.get("title"),
            "main_category":  r.get("main_category"),
            "store":          r.get("store"),
            "price":          r.get("price"),
            "average_rating": r.get("average_rating"),
            "rating_number":  r.get("rating_number"),
            "embedding_text": r.get("embedding_text"),
            "full_text":      r.get("full_text"),
        }
        for i, r in enumerate(records)
    ]


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main() -> None:
    if not INPUT_RECORDS_PATH.exists():
        log.error("Input not found: %s", INPUT_RECORDS_PATH)
        sys.exit(1)

    log.info("Loading records from %s", INPUT_RECORDS_PATH)
    records = load_jsonl(INPUT_RECORDS_PATH)
    log.info("Loaded %d records", len(records))

    records = filter_records(records)
    if not records:
        log.error("No valid records after filtering — aborting.")
        sys.exit(1)

    vecs = embed_all(records)

    index = build_index(vecs)
    INDEX_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_OUTPUT_PATH))
    log.info("FAISS index saved → %s", INDEX_OUTPUT_PATH)

    metadata = build_metadata(records)
    save_jsonl(metadata, METADATA_OUTPUT_PATH)

    log.info("Done.  Index size: %d  |  Dim: %d", index.ntotal, vecs.shape[1])


if __name__ == "__main__":
    main()
