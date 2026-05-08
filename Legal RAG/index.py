"""
Supreme Court Judgments — Vector + BM25 Indexing Pipeline (Stage 3)
====================================================================

Reads the output of embed_chunks.py (embeddings.npy + chunks.jsonl) and builds
two complementary indexes that together support hybrid retrieval:

    1. Qdrant dense index  — cosine similarity over L2-normalised float32 vectors.
         Stored in a Qdrant server running in Docker and reached via HTTP
         (default: http://localhost:6333).

  2. BM25 sparse index   — BM25Okapi over lowercased word-tokens.
     Stored as <output-dir>/bm25_index.pkl.

  3. Chunk store         — <output-dir>/chunk_store.pkl  {"texts", "metadatas", "ids"}
     Loaded at retrieval time to hydrate BM25 hits with original text + metadata.

  4. Index manifest      — <output-dir>/index_manifest.json  (paths, counts, timing).

After indexing, a smoke test runs 3 representative queries through both indexes
and prints the top-3 results from each so you can sanity-check retrieval quality
before building the retrieval layer.

Usage:
    python index.py                                   # use defaults
    python index.py --embeddings-dir embeddings_arctic_m_v2 --dry-run
    python index.py --output-dir my_index --qdrant-batch 256
    docker compose up -d qdrant                       # start Qdrant server
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

if os.name == "nt":
    # Workaround for duplicate OpenMP runtime initialisation on some Windows envs.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

# ──────────────────────────────────────────────
# Defaults  (all resolved relative to this file)
# ──────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent

DEFAULT_EMBEDDINGS_DIR = _HERE / "embeddings_arctic_m_v2"
DEFAULT_CHUNKS_JSONL   = _HERE / "chunks.jsonl"
DEFAULT_OUTPUT_DIR     = _HERE / "index_output"
DEFAULT_QDRANT_BATCH   = 500
DEFAULT_QDRANT_URL     = "http://localhost:6333"
QDRANT_COLLECTION      = "sc_judgments"
LOCAL_MODELS_ROOT      = _HERE / ".hf_models"

SMOKE_TEST_QUERIES = [
    "right to life Article 21 fundamental rights",
    "AIR 1978 SC 597 Maneka Gandhi",
    "anticipatory bail non-bailable offence",
]


# ──────────────────────────────────────────────
# Logging  (file + console, matching project style)
# ──────────────────────────────────────────────
def _setup_logging(log_file: Path) -> logging.Logger:
    """Configure module-level logger that writes to both console and *log_file*."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments for the indexing pipeline."""
    parser = argparse.ArgumentParser(
        description="Build Qdrant + BM25 indexes from embed_chunks.py output."
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR,
        help="Folder containing embeddings.npy and embedding_manifest.json "
             "(default: ./embeddings_arctic_m_v2)",
    )
    parser.add_argument(
        "--chunks-jsonl",
        type=Path,
        default=DEFAULT_CHUNKS_JSONL,
        help="Path to chunks.jsonl produced by preprocess_and_chunk.py "
             "(default: ./chunks.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for all index outputs (default: ./index_output)",
    )
    parser.add_argument(
        "--qdrant-batch",
        type=int,
        default=DEFAULT_QDRANT_BATCH,
        help="Number of PointStruct objects per Qdrant upsert call (default: 500)",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=DEFAULT_QDRANT_URL,
        help="HTTP URL for the Qdrant server (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=os.environ.get("QDRANT_API_KEY", ""),
        help="Optional API key for Qdrant server auth; defaults to QDRANT_API_KEY",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Index only the first 5 000 chunks to verify the pipeline end-to-end",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# 1. Input loaders
# ──────────────────────────────────────────────
def load_embedding_manifest(embeddings_dir: Path) -> dict[str, Any]:
    """
    Read embedding_manifest.json from *embeddings_dir* and return it as a dict.

    Raises FileNotFoundError with a clear message when the file is absent so
    callers can surface an actionable error rather than a raw Python traceback.
    """
    manifest_path = embeddings_dir / "embedding_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"embedding_manifest.json not found in {embeddings_dir}. "
            "Run embed_chunks.py first to generate embeddings."
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings(embeddings_dir: Path) -> np.ndarray:
    """
    Memory-map embeddings.npy from *embeddings_dir* and return a float32 array
    of shape (N, embedding_dim).

    Using memory-mapping avoids loading the entire file into RAM at once,
    which matters for corpora with millions of chunks.
    """
    emb_path = embeddings_dir / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"embeddings.npy not found in {embeddings_dir}. "
            "Run embed_chunks.py first to generate embeddings."
        )
    return np.load(str(emb_path), mmap_mode="r").astype(np.float32)


def iter_chunks(chunks_jsonl: Path) -> Iterator[dict[str, Any]]:
    """
    Stream chunk objects from *chunks_jsonl* one at a time without loading the
    entire file into memory.

    Each yielded dict has the shape written by preprocess_and_chunk.py:
        { "chunk_id": str, "text": str, "metadata": { ... } }
    """
    if not chunks_jsonl.exists():
        raise FileNotFoundError(
            f"chunks.jsonl not found at {chunks_jsonl}. "
            "Run preprocess_and_chunk.py first."
        )
    with chunks_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_no} in {chunks_jsonl}: {exc}"
                ) from exc


def load_all_chunks(
    chunks_jsonl: Path,
    limit: int | None = None,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """
    Load all chunks from *chunks_jsonl* and return three parallel lists:
        texts     — raw chunk text strings
        metadatas — full metadata dicts (case_title, year, …)
        ids       — chunk_id strings

    *limit* is used by --dry-run to cap the number of chunks loaded.
    Keeping the three lists index-aligned is essential: the i-th text, the i-th
    metadata, and the i-th id all correspond to the same chunk, which in turn
    corresponds to row i of the embeddings matrix.
    """
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    for chunk in iter_chunks(chunks_jsonl):
        texts.append(chunk["text"])
        metadatas.append(chunk.get("metadata", {}))
        ids.append(chunk["chunk_id"])
        if limit is not None and len(texts) >= limit:
            break

    return texts, metadatas, ids


# ──────────────────────────────────────────────
# 2. Qdrant dense index
# ──────────────────────────────────────────────
def _collection_point_count(client: QdrantClient, collection_name: str) -> int:
    """
    Return the number of indexed points in *collection_name*, or -1 if the
    collection does not exist yet.

    Qdrant raises an exception (rather than returning None) when a collection
    is missing, so we use a try/except to implement this existence check.
    """
    try:
        info = client.get_collection(collection_name)
        return info.points_count or 0
    except Exception:
        return -1


def create_qdrant_client(qdrant_url: str, qdrant_api_key: str = "") -> QdrantClient:
    """
    Create an HTTP Qdrant client for a server running outside the Python process.

    This avoids embedded local mode, which is not suitable for collections at the
    scale of this corpus.  The default deployment target is the Docker Compose
    service in the workspace root, exposed on http://localhost:6333.
    """
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
        client.get_collections()
        return client
    except Exception as exc:
        raise RuntimeError(
            f"Unable to connect to Qdrant at {qdrant_url}. Start the Docker service "
            "with 'docker compose up -d qdrant' from the workspace root, then retry. "
            f"Original error: {exc}"
        ) from exc


def build_qdrant_index(
    embeddings: np.ndarray,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict[str, Any]],
    qdrant_url: str,
    qdrant_api_key: str = "",
    collection_name: str = QDRANT_COLLECTION,
    batch_size: int = DEFAULT_QDRANT_BATCH,
) -> QdrantClient:
    """
    Upsert all vectors and payloads into a Docker-backed Qdrant collection.

    Design decisions:
    - QdrantClient(url=…) talks to a standalone Qdrant server running in Docker,
      which avoids the performance and scale limits of embedded local mode.
    - Vectors are already L2-normalised by embed_chunks.py, so cosine distance
      is equivalent to dot-product and gives exact nearest-neighbour ordering.
    - Skip re-creation when the collection already contains the expected number
      of points so the pipeline is resumable without re-uploading everything.
    - Each PointStruct payload carries all fields needed for retrieval without
      a secondary lookup into the chunk store, making Qdrant results self-contained.
    """
    client = create_qdrant_client(qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key)

    total = len(ids)
    existing = _collection_point_count(client, collection_name)

    if existing == total:
        logger.info(
            "Qdrant collection '%s' already has %d points — skipping re-index.",
            collection_name, total,
        )
        return client

    if existing > 0:
        logger.info(
            "Qdrant collection '%s' has %d/%d points — recreating from scratch.",
            collection_name, existing, total,
        )
        client.delete_collection(collection_name)

    embedding_dim = embeddings.shape[1]
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )
    logger.info(
        "Created Qdrant collection '%s'  (dim=%d, distance=COSINE).",
        collection_name, embedding_dim,
    )

    uploaded = 0
    with tqdm(total=total, desc="Qdrant upsert", unit="vec") as pbar:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            points: list[PointStruct] = []

            for i in range(batch_start, batch_end):
                meta = metadatas[i]
                payload: dict[str, Any] = {
                    "chunk_id": ids[i],
                    "text": texts[i],
                    "case_title": meta.get("case_title", ""),
                    "year": meta.get("year", 0),
                    "petitioner": meta.get("petitioner", ""),
                    "respondent": meta.get("respondent", ""),
                    "judgment_date": meta.get("judgment_date", ""),
                    "chunk_index": meta.get("chunk_index", i),
                    "total_chunks": meta.get("total_chunks", total),
                }
                points.append(
                    PointStruct(
                        id=i,
                        vector=embeddings[i].tolist(),
                        payload=payload,
                    )
                )

            try:
                client.upsert(collection_name=collection_name, points=points)
            except Exception as exc:
                logger.error(
                    "Qdrant upsert failed for batch %d–%d: %s",
                    batch_start, batch_end - 1, exc,
                )
                raise

            uploaded += len(points)
            pbar.update(len(points))

            if (batch_start // batch_size) % 10 == 0:
                logger.info(
                    "Qdrant: uploaded %d / %d vectors (batch %d–%d).",
                    uploaded, total, batch_start, batch_end - 1,
                )

    logger.info("Qdrant indexing complete — %d vectors indexed.", uploaded)
    return client


# ──────────────────────────────────────────────
# 3. BM25 sparse index
# ──────────────────────────────────────────────
def build_bm25_index(texts: list[str]) -> BM25Okapi:
    """
    Build a BM25Okapi index from *texts* using simple whitespace tokenisation.

    rank_bm25's BM25Okapi accepts a list of token lists and pre-computes the
    IDF weights and document-length statistics needed for scoring at retrieval
    time.  We use lower-cased whitespace splitting to stay lightweight and avoid
    a tokenizer dependency at retrieval time; the same logic must be applied to
    query strings when searching.
    """
    t0 = time.time()
    logger.info("Tokenising %d chunks for BM25 …", len(texts))
    tokenised = [t.lower().split() for t in texts]
    logger.info("Building BM25Okapi index …")
    bm25 = BM25Okapi(tokenised)
    logger.info("BM25 index built in %.1f s.", time.time() - t0)
    return bm25


def save_bm25_index(bm25: BM25Okapi, path: Path) -> None:
    """Pickle *bm25* to *path* so it can be restored without re-tokenising."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    _log_file_size(path, "BM25 index")


def save_chunk_store(
    texts: list[str],
    metadatas: list[dict[str, Any]],
    ids: list[str],
    path: Path,
) -> None:
    """
    Pickle the chunk store dict to *path*.

    The chunk store is a flat dict with three index-aligned lists so any
    retrieval layer can hydrate BM25 hits with text and metadata using a
    simple integer index lookup, without touching the original JSONL file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    store = {"texts": texts, "metadatas": metadatas, "ids": ids}
    with path.open("wb") as f:
        pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
    _log_file_size(path, "Chunk store")


# ──────────────────────────────────────────────
# 4. Index manifest
# ──────────────────────────────────────────────
def save_index_manifest(
    output_dir: Path,
    qdrant_url: str,
    bm25_path: Path,
    chunk_store_path: Path,
    num_chunks: int,
    embedding_dim: int,
    model_name: str,
    elapsed: float,
) -> None:
    """
    Write index_manifest.json to *output_dir* summarising all index artefacts.

    The manifest is consumed by the retrieval layer (query.py) so it can locate
    all index files without hard-coded paths or environment variables.
    """
    manifest = {
        "qdrant_collection":  QDRANT_COLLECTION,
        "qdrant_mode":        "docker",
        "qdrant_url":         qdrant_url,
        "bm25_path":          str(bm25_path.resolve()),
        "chunk_store_path":   str(chunk_store_path.resolve()),
        "num_chunks":         num_chunks,
        "embedding_dim":      embedding_dim,
        "model_name":         model_name,
        "indexed_at":         datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds":    round(elapsed, 2),
    }
    manifest_path = output_dir / "index_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("Index manifest saved to %s", manifest_path)


# ──────────────────────────────────────────────
# 5. Query encoder (mirrors embed_chunks.py; query prefix only at query time)
# ──────────────────────────────────────────────
def load_query_encoder(
    model_name: str,
) -> tuple[AutoTokenizer, AutoModel, torch.device]:
    """
    Load the tokenizer and model used to encode search queries at smoke-test time.

    This intentionally reuses the same local model snapshot that embed_chunks.py
    wrote to .hf_models/<model-slug>/ so no additional network download is needed.
    Documents were encoded without any prefix; queries must be prefixed with
    "query: " as required by arctic-embed's asymmetric retrieval setup.

    The model is loaded in eval mode and moved to CUDA if available, falling back
    to CPU so the smoke test runs even on machines without a GPU.
    """
    local_model_dir = LOCAL_MODELS_ROOT / model_name.replace("/", "--")
    if not local_model_dir.exists():
        raise FileNotFoundError(
            f"Local model not found at {local_model_dir}. "
            "Run embed_chunks.py first so the model snapshot is downloaded."
        )

    base_kwargs: dict[str, Any] = {"trust_remote_code": True, "local_files_only": True}

    config = AutoConfig.from_pretrained(str(local_model_dir), **base_kwargs)
    # arctic-embed may request memory-efficient attention; disable for portability.
    for attr in ("use_memory_efficient_attention", "memory_efficient_attention"):
        if hasattr(config, attr):
            setattr(config, attr, False)
    if hasattr(config, "attn_implementation"):
        setattr(config, "attn_implementation", "eager")

    tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir), **base_kwargs)
    model = AutoModel.from_pretrained(str(local_model_dir), config=config, **base_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info("Query encoder loaded on %s from %s", device, local_model_dir)
    return tokenizer, model, device


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool *hidden* states weighted by the attention *mask*.

    Identical to the pooling used in embed_chunks.py so query and document
    vectors live in the same space.
    """
    expanded = mask.unsqueeze(-1).expand(hidden.size()).float()
    summed = torch.sum(hidden * expanded, dim=1)
    counts = torch.clamp(expanded.sum(dim=1), min=1e-9)
    return summed / counts


def encode_query(
    query_text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int = 512,
) -> np.ndarray:
    """
    Encode a single *query_text* into an L2-normalised float32 vector.

    The "query: " prefix is prepended here — at query time — never during
    document indexing.  This asymmetric encoding is the recommended usage for
    Snowflake arctic-embed models: documents are stored without prefix and
    queries are distinguished by the prefix so the model can specialise its
    representations accordingly.
    """
    prefixed = f"query: {query_text}"
    encoded = tokenizer(
        [prefixed],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.inference_mode():
        outputs = model(**encoded)
        pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

    return pooled.detach().cpu().numpy().astype(np.float32)[0]


def query_qdrant_points(
    client: QdrantClient,
    collection_name: str,
    query_vector: np.ndarray,
    limit: int,
) -> list[Any]:
    """
    Query *collection_name* with *query_vector* and return scored hits.

    qdrant-client changed its high-level search API across releases: older
    versions exposed ``client.search(...)`` while newer releases route vector
    retrieval through ``client.query_points(...)``.  This helper hides that
    version difference so the smoke test works against both interfaces without
    pinning the project to one exact client version.
    """
    vector = query_vector.tolist()

    if hasattr(client, "search"):
        return client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
        )

    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=limit,
    )
    if hasattr(response, "points"):
        return list(response.points)
    return list(response)


# ──────────────────────────────────────────────
# 5b. Smoke test
# ──────────────────────────────────────────────
def run_smoke_tests(
    client: QdrantClient,
    bm25: BM25Okapi,
    chunk_store: dict[str, Any],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    collection_name: str = QDRANT_COLLECTION,
    top_k: int = 10,
    display_k: int = 3,
) -> None:
    """
    Run SMOKE_TEST_QUERIES through both BM25 and Qdrant and print top results.

    Purpose: verify end-to-end retrieval quality immediately after indexing so
    any mis-configuration (wrong prefix, wrong collection, misaligned chunk store)
    is caught before the retrieval layer is built on top of these indexes.

    Each query prints *display_k* results from each index showing:
        rank, score, case_title, year, chunk_id
    """
    texts     = chunk_store["texts"]
    metadatas = chunk_store["metadatas"]
    ids       = chunk_store["ids"]

    separator = "─" * 72
    print(f"\n{'═' * 72}")
    print("  SMOKE TEST — verifying retrieval quality")
    print(f"{'═' * 72}")

    for q_idx, query in enumerate(SMOKE_TEST_QUERIES, start=1):
        print(f"\n[{q_idx}] Query: \"{query}\"")
        print(separator)

        # ── BM25 ─────────────────────────────────────────────────────────────
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)
        top_bm25_indices = np.argsort(scores)[::-1][:top_k]

        print(f"  BM25 top-{display_k}:")
        for rank, idx in enumerate(top_bm25_indices[:display_k], start=1):
            meta = metadatas[idx]
            print(
                f"    #{rank}  score={scores[idx]:.4f}  "
                f"year={meta.get('year', '?')}  "
                f"case={meta.get('case_title', 'N/A')[:55]}  "
                f"id={ids[idx]}"
            )

        # ── Qdrant ───────────────────────────────────────────────────────────
        q_vec = encode_query(query, tokenizer, model, device)
        hits = query_qdrant_points(
            client=client,
            collection_name=collection_name,
            query_vector=q_vec,
            limit=top_k,
        )

        print(f"  Qdrant top-{display_k}:")
        for rank, hit in enumerate(hits[:display_k], start=1):
            payload = hit.payload or {}
            print(
                f"    #{rank}  score={hit.score:.4f}  "
                f"year={payload.get('year', '?')}  "
                f"case={str(payload.get('case_title', 'N/A'))[:55]}  "
                f"id={payload.get('chunk_id', hit.id)}"
            )

    print(f"\n{'═' * 72}\n")


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────
def _log_file_size(path: Path, label: str) -> None:
    """Log the size of *path* in a human-readable unit (KB / MB / GB)."""
    if not path.exists():
        return
    size_bytes = path.stat().st_size
    if size_bytes >= 1_073_741_824:
        logger.info("%s: %.2f GB  (%s)", label, size_bytes / 1_073_741_824, path)
    elif size_bytes >= 1_048_576:
        logger.info("%s: %.1f MB  (%s)", label, size_bytes / 1_048_576, path)
    else:
        logger.info("%s: %.1f KB  (%s)", label, size_bytes / 1_024, path)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main() -> None:
    """Orchestrate the full indexing pipeline and smoke test."""
    args = parse_args()

    # Set up logging now that we know the output directory.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.output_dir / "index.log")

    pipeline_start = time.time()
    dry_run_limit = 5_000 if args.dry_run else None

    if args.dry_run:
        logger.info("--dry-run: indexing only the first %d chunks.", dry_run_limit)

    # ── Step 1: load inputs ──────────────────────────────────────────────────
    logger.info("Loading embedding manifest from %s …", args.embeddings_dir)
    manifest = load_embedding_manifest(args.embeddings_dir)
    model_name: str = manifest["model_name"]
    embedding_dim: int = manifest["embedding_dim"]
    logger.info(
        "Manifest: model=%s  dim=%d  num_chunks=%d",
        model_name, embedding_dim, manifest["num_chunks"],
    )

    logger.info("Memory-mapping embeddings from %s …", args.embeddings_dir)
    embeddings = load_embeddings(args.embeddings_dir)
    if dry_run_limit:
        embeddings = embeddings[:dry_run_limit]
    logger.info("Embeddings shape: %s", embeddings.shape)

    logger.info("Loading chunks from %s …", args.chunks_jsonl)
    texts, metadatas, ids = load_all_chunks(args.chunks_jsonl, limit=dry_run_limit)
    num_chunks = len(texts)
    logger.info("Loaded %d chunks.", num_chunks)

    if num_chunks != embeddings.shape[0]:
        raise ValueError(
            f"Chunk count ({num_chunks}) does not match embedding rows "
            f"({embeddings.shape[0]}). Ensure chunks.jsonl and embeddings.npy "
            "were generated from the same run."
        )

    # ── Step 2: Qdrant dense index ───────────────────────────────────────────
    logger.info("Building Qdrant index at %s …", args.qdrant_url)
    qdrant_t0 = time.time()
    client = build_qdrant_index(
        embeddings=embeddings,
        ids=ids,
        texts=texts,
        metadatas=metadatas,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=QDRANT_COLLECTION,
        batch_size=args.qdrant_batch,
    )
    logger.info("Qdrant indexing finished in %.1f s.", time.time() - qdrant_t0)

    # ── Step 3: BM25 sparse index ────────────────────────────────────────────
    bm25_path       = args.output_dir / "bm25_index.pkl"
    chunk_store_path = args.output_dir / "chunk_store.pkl"

    logger.info("Building BM25 index …")
    bm25 = build_bm25_index(texts)
    save_bm25_index(bm25, bm25_path)
    save_chunk_store(texts, metadatas, ids, chunk_store_path)

    # ── Step 4: index manifest ───────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    save_index_manifest(
        output_dir=args.output_dir,
        qdrant_url=args.qdrant_url,
        bm25_path=bm25_path,
        chunk_store_path=chunk_store_path,
        num_chunks=num_chunks,
        embedding_dim=embedding_dim,
        model_name=model_name,
        elapsed=elapsed,
    )

    logger.info(
        "All indexes saved to %s  |  elapsed: %.1f s",
        args.output_dir, elapsed,
    )

    # ── Step 5: smoke test ───────────────────────────────────────────────────
    logger.info("Loading query encoder for smoke test …")
    tokenizer, enc_model, device = load_query_encoder(model_name)

    chunk_store: dict[str, Any] = {"texts": texts, "metadatas": metadatas, "ids": ids}
    run_smoke_tests(
        client=client,
        bm25=bm25,
        chunk_store=chunk_store,
        tokenizer=tokenizer,
        model=enc_model,
        device=device,
    )

    logger.info("index.py complete — total elapsed: %.1f s", time.time() - pipeline_start)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
