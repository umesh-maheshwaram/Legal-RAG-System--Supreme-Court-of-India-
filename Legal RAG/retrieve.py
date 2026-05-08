"""
retrieve.py — Stage 4: Retrieval + LLM Answer Generation
Supreme Court Judgments RAG Pipeline
 
This is the final stage of the pipeline. It wires together
everything built in Stages 1–3 into a single query interface.
 
Pipeline per query:
  1. Encode query  → arctic-embed-m-v2.0 (768-dim vector, mean-pooled,
                     "query: " prefix, matching embed_chunks.py exactly)
  2. Dense search  → Qdrant HNSW (top-50 semantic matches)
  3. Sparse search → BM25 (top-50 keyword/citation matches)
  4. Merge + dedup → RRF fusion (Reciprocal Rank Fusion, k=60)
  5. Rerank        → bge-reranker-v2-m3 CrossEncoder (top-5)
  6. Build context → structured prompt from top-5 chunks
  7. Generate      → SaulLM-7B fine-tuned (saullm-legal via Ollama)
  8. Return        → answer text + source citations + latency
 
 
Inputs (all produced by earlier stages):
  index_output/
    qdrant_db/          Stage 3 — Qdrant Docker (localhost:6333)
    bm25_index.pkl      Stage 3 — BM25 sparse index
    chunk_store.pkl     Stage 3 — texts + metadatas + chunk_ids
    index_manifest.json Stage 3 — index metadata
  .hf_models/
    Snowflake--snowflake-arctic-embed-m-v2.0/   Stage 2 model
  saullm_indic_legal/   Stage 4 — fine-tuned LoRA adapter
  OR
  Ollama running saullm-legal model
 
Hardware : NVIDIA RTX A4000 16 GB | i9-14900K | 64 GB RAM
           Windows 11 | conda env: rag
 
Usage:
  # Interactive mode
  python retrieve.py
 
  # Single query
  python retrieve.py --query "What is Article 21 right to life?"
 
  # Citation lookup
  python retrieve.py --query "AIR 1978 SC 597 Maneka Gandhi"
 
  # Use Ollama backend (recommended after merge + GGUF)
  python retrieve.py --llm-backend ollama --ollama-model saullm-legal
 
  # Use HuggingFace adapter directly (no Ollama needed)
  python retrieve.py --llm-backend hf
 
  # Disable LLM — retrieval only (fast, no GPU for generation)
  python retrieve.py --llm-backend none --query "anticipatory bail"
"""
 
# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────
import gc
import json
import logging
import os
import pickle
import sys
import time
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
 
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
import requests
 
 
# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent
INDEX_DIR       = ROOT_DIR / "index_output"
BM25_PATH       = INDEX_DIR / "bm25_index.pkl"
CHUNK_STORE     = INDEX_DIR / "chunk_store.pkl"
INDEX_MANIFEST  = INDEX_DIR / "index_manifest.json"
LOG_FILE        = ROOT_DIR / "retrieve.log"
QUERY_LOG       = ROOT_DIR / "query_log.jsonl"
 
# Embedding model — same as used in Stage 2
HF_MODELS_DIR   = ROOT_DIR / ".hf_models"
EMBED_MODEL_DIR = HF_MODELS_DIR / "Snowflake--snowflake-arctic-embed-m-v2.0"
 
# Fine-tuned LoRA adapter — from Stage 4 fine-tuning
ADAPTER_DIR     = ROOT_DIR / "saullm_indic_legal"
BASE_MODEL_DIR  = HF_MODELS_DIR / "Saul-7B-Base"
 
 
# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
 
# Retrieval config
DENSE_TOP_K         = 50       # candidates from Qdrant
SPARSE_TOP_K        = 50       # candidates from BM25
RERANK_TOP_K        = 5        # final chunks passed to LLM
MAX_SEQ_LENGTH      = 512      # embedding model token limit
 
# RRF fusion config
# k=60 is the standard constant from the original RRF paper (Cormack 2009).
# Higher k reduces the weight of top-ranked results; 60 works well in practice.
RRF_K               = 60
 
# Qdrant Docker config
QDRANT_HOST         = "localhost"
QDRANT_PORT         = 6333
QDRANT_COLLECTION   = "sc_judgments"
 
# Reranker model
RERANKER_MODEL      = "BAAI/bge-reranker-v2-m3"
 
# LLM config
OLLAMA_URL          = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL= "saullm-legal"   # after GGUF conversion
LLM_MAX_NEW_TOKENS  = 400
LLM_TEMPERATURE     = 0.1
 
# System prompt — MUST match the format used in fine-tuning
SYSTEM_PROMPT = (
    "You are a senior legal assistant specializing in Indian Supreme Court law.\n\n"
    "Answer strictly based on the provided case excerpts.\n\n"
    "Your answer MUST follow this structure:\n"
    "1. Legal Principle\n"
    "2. Leading Case(s) with year\n"
    "3. Explanation (clear and precise)\n\n"
    "Do NOT give vague or generic answers.\n"
    "Do NOT hallucinate.\n"
    "Cite case names explicitly.\n"
)
 
 
# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
 
 
# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    chunk_id        : str
    text            : str
    case_title      : str
    year            : int
    petitioner      : str
    respondent      : str
    judgment_date   : str
    chunk_index     : int
    dense_score     : float = 0.0
    sparse_score    : float = 0.0
    rerank_score    : float = 0.0
    rrf_score       : float = 0.0   # [FIX 2] RRF fusion score
    source          : str  = ""
 
 
@dataclass
class QueryResult:
    query           : str
    answer          : str
    sources         : list[dict]
    latency_ms      : dict
    total_ms        : float
    llm_backend     : str
 
 
# ─────────────────────────────────────────────
# Main retriever class
# ─────────────────────────────────────────────
class LegalRetriever:
 
    def __init__(
        self,
        llm_backend     : str  = "ollama",
        ollama_model    : str  = DEFAULT_OLLAMA_MODEL,
        qdrant_host     : str  = QDRANT_HOST,
        qdrant_port     : int  = QDRANT_PORT,
    ):
        self.llm_backend    = llm_backend
        self.ollama_model   = ollama_model
        self.qdrant_host    = qdrant_host
        self.qdrant_port    = qdrant_port
        self.device         = self._get_device()
 
        self.embed_model        = None
        self.embed_tokenizer    = None
        self.reranker_model     = None
        self.reranker_tokenizer = None
        self.qdrant_client      = None
        self.bm25               = None
        self.texts              = []
        self.metadatas          = []
        self.chunk_ids          = []
        self.saul_model         = None
        self.saul_tokenizer     = None
 
        self._load_all()
 
    # ──────────────────────────────────────────
    # Setup helpers
    # ──────────────────────────────────────────
    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("GPU  : %s  (%.1f GB VRAM)", name, vram)
            return torch.device("cuda")
        logger.warning("No GPU — running on CPU. Retrieval will be slower.")
        return torch.device("cpu")
 
    def _load_all(self) -> None:
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("Loading retrieval pipeline components...")
 
        self._load_embed_model()
        self._load_reranker()
        self._load_qdrant()
        self._load_bm25_and_store()
 
        # [FIX 3] Run ID mismatch diagnostic immediately after both
        # indexes are loaded so any chunk_id format divergence is
        # caught before the first query is ever issued.
        self._debug_id_mismatch()
 
        if self.llm_backend == "hf":
            self._load_saullm_hf()
        elif self.llm_backend == "ollama":
            self._check_ollama()
        else:
            logger.info("LLM backend: none — retrieval only mode")
 
        logger.info("Pipeline ready  |  elapsed: %.1f s", time.time() - t0)
        logger.info("=" * 60)
 
    def _load_embed_model(self) -> None:
        """
        Load arctic-embed-m-v2.0 for query encoding.
 
        Patches the config to disable xformers memory-efficient attention
        before loading — avoids AssertionError crash when xformers is
        not installed.
        """
        if not EMBED_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Embedding model not found: {EMBED_MODEL_DIR}\n"
                "Expected: .hf_models/Snowflake--snowflake-arctic-embed-m-v2.0/"
            )
 
        logger.info("Loading query encoder: %s", EMBED_MODEL_DIR.name)
 
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            str(EMBED_MODEL_DIR),
            trust_remote_code=True,
        )
        if hasattr(config, "use_memory_efficient_attention"):
            config.use_memory_efficient_attention = False
        if hasattr(config, "memory_efficient_attention"):
            config.memory_efficient_attention = False
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "eager"
 
        logger.info(
            "Patched config: use_memory_efficient_attention=False "
            "(xformers not required)"
        )
 
        self.embed_tokenizer = AutoTokenizer.from_pretrained(
            str(EMBED_MODEL_DIR),
            trust_remote_code=True,
        )
        self.embed_model = AutoModel.from_pretrained(
            str(EMBED_MODEL_DIR),
            config=config,
            add_pooling_layer=False,
            trust_remote_code=True,
            dtype=torch.float16,
        )
        self.embed_model.eval()
        self.embed_model.to(self.device)
 
        logger.info(
            "Query encoder ready  |  device=%s  |  dtype=%s",
            self.device,
            next(self.embed_model.parameters()).dtype,
        )
 
    def _load_reranker(self) -> None:
        """
        Load bge-reranker-v2-m3 via AutoModelForSequenceClassification.
 
        Bypasses CrossEncoder wrapper which fails with model_type ValueError
        on this model version. Direct loading is identical internally.
        """
        from transformers import AutoModelForSequenceClassification
        from huggingface_hub import snapshot_download
 
        reranker_local = HF_MODELS_DIR / "BAAI--bge-reranker-v2-m3"
 
        if not reranker_local.exists() or not (reranker_local / "config.json").exists():
            logger.info("Downloading bge-reranker-v2-m3 to: %s", reranker_local)
            snapshot_download(
                repo_id="BAAI/bge-reranker-v2-m3",
                local_dir=str(reranker_local),
                ignore_patterns=["onnx/*", "*.ot"],
            )
            logger.info("Reranker download complete")
        else:
            logger.info("Loading reranker from local: %s", reranker_local.name)
 
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            str(reranker_local),
            trust_remote_code=True,
        )
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            str(reranker_local),
            trust_remote_code=True,
            dtype=torch.float16,
            num_labels=1,
        )
        self.reranker_model.eval()
        self.reranker_model.to(self.device)
        logger.info(
            "Reranker ready  |  device=%s  |  dtype=%s",
            self.device,
            next(self.reranker_model.parameters()).dtype,
        )
 
    def _load_qdrant(self) -> None:
        logger.info(
            "Connecting to Qdrant at %s:%d",
            self.qdrant_host, self.qdrant_port
        )
 
        try:
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port,
            )
            collections = [
                c.name for c in
                self.qdrant_client.get_collections().collections
            ]
            if QDRANT_COLLECTION not in collections:
                raise RuntimeError(
                    f"Collection '{QDRANT_COLLECTION}' not found in Qdrant.\n"
                    "Run index.py first to build the collection."
                )
 
            info  = self.qdrant_client.get_collection(QDRANT_COLLECTION)
            count = info.points_count
 
            # [FIX 4] Log the Qdrant vector config so we can confirm
            # 768-dim COSINE collection was created correctly by index.py.
            logger.info(
                "Qdrant connected  |  collection=%s  |  points=%s",
                QDRANT_COLLECTION,
                f"{count:,}"
            )
            logger.info(
                "Qdrant vector config: %s",
                info.config.params.vectors,
            )
 
        except Exception as exc:
            if "Connection refused" in str(exc) or "connect" in str(exc).lower():
                raise RuntimeError(
                    "Cannot connect to Qdrant at localhost:6333.\n"
                    "Start Qdrant Docker with:\n"
                    "  docker start sc-qdrant\n"
                    "Or if first time:\n"
                    "  docker run -d --name sc-qdrant \\\n"
                    "    -p 6333:6333 -p 6334:6334 \\\n"
                    "    -v C:/Umesh/project_RAG/qdrant_storage:/qdrant/storage \\\n"
                    "    qdrant/qdrant"
                ) from exc
            raise
 
    def _load_bm25_and_store(self) -> None:
        for path, label in [
            (BM25_PATH,   "BM25 index"),
            (CHUNK_STORE, "Chunk store"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{label} not found: {path}\n"
                    "Run index.py first."
                )
 
        logger.info("Loading BM25 index (~1.09 GB)...")
        with open(BM25_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
 
        logger.info("Loading chunk store (~1.47 GB)...")
        with open(CHUNK_STORE, "rb") as f:
            store = pickle.load(f)
 
        self.texts      = store["texts"]
        self.metadatas  = store["metadatas"]
        self.chunk_ids  = store["ids"]
 
        logger.info(
            "Indexes loaded  |  chunks=%s  |  BM25 vocab=%s",
            f"{len(self.texts):,}",
            f"{len(self.bm25.idf):,}" if hasattr(self.bm25, 'idf') else "N/A",
        )
 
    def _load_saullm_hf(self) -> None:
        from transformers import BitsAndBytesConfig, AutoModelForCausalLM
        from peft import PeftModel
 
        if not BASE_MODEL_DIR.exists():
            raise FileNotFoundError(f"SaulLM base model not found: {BASE_MODEL_DIR}")
        if not ADAPTER_DIR.exists():
            raise FileNotFoundError(
                f"Fine-tuned adapter not found: {ADAPTER_DIR}\n"
                "Run finetune.py first."
            )
 
        logger.info("Loading SaulLM-7B (4-bit) + fine-tuned adapter...")
 
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
 
        self.saul_tokenizer = AutoTokenizer.from_pretrained(
            str(BASE_MODEL_DIR),
            trust_remote_code=True,
        )
        if self.saul_tokenizer.pad_token is None:
            self.saul_tokenizer.pad_token = self.saul_tokenizer.eos_token
 
        base = AutoModelForCausalLM.from_pretrained(
            str(BASE_MODEL_DIR),
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
            dtype=torch.float16,
        )
 
        self.saul_model = PeftModel.from_pretrained(
            base,
            str(ADAPTER_DIR),
            dtype=torch.float16,
        )
        self.saul_model.eval()
        logger.info("SaulLM-7B loaded with fine-tuned adapter")
 
    def _check_ollama(self) -> None:
        logger.info(
            "Checking Ollama at %s with model '%s'",
            OLLAMA_URL, self.ollama_model
        )
 
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
 
            model_found = any(self.ollama_model in m for m in models)
 
            if not model_found:
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s\n"
                    "Create it with: ollama create saullm-legal -f Modelfile",
                    self.ollama_model,
                    models[:5]
                )
            else:
                logger.info("Ollama ready  |  model=%s", self.ollama_model)
 
        except requests.exceptions.ConnectionError:
            logger.warning(
                "Cannot connect to Ollama at %s\n"
                "Start Ollama: open Ollama application or run 'ollama serve'\n"
                "LLM generation will fail but retrieval will still work.",
                OLLAMA_URL
            )
 
    # ──────────────────────────────────────────
    # [FIX 3] ID mismatch diagnostic
    # ──────────────────────────────────────────
    def _debug_id_mismatch(self) -> None:
        """
        Compare chunk_ids from Qdrant payload vs chunk_store to catch
        any format divergence that would cause both=0 in merge.
 
        index.py stores chunk_id in the Qdrant payload (payload["chunk_id"])
        and also in chunk_store["ids"]. Both should be strings of the form:
            "2020/Parvez_Noordin_Lokhandwalla_vs_....__chunk_0042"
 
        If the formats differ (e.g. one strips the year prefix), no chunk
        will ever appear in both dense and sparse results → both=0 forever.
 
        This runs once at startup. Check the log output for:
            [ID CHECK] Qdrant payload chunk_id : 2020/Case_Name__chunk_0001
            [ID CHECK] chunk_store sample IDs  : ['2020/Case_Name__chunk_0001', ...]
        They must look identical for hybrid merge to work.
        """
        logger.info("=" * 60)
        logger.info("[ID CHECK] Verifying Qdrant ↔ chunk_store ID alignment...")
        try:
            sample_points, _ = self.qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=3,
                with_payload=True,
            )
            logger.info("[ID CHECK] Qdrant point samples:")
            for pt in sample_points:
                payload_chunk_id = pt.payload.get("chunk_id", "MISSING")
                raw_point_id     = str(pt.id)
                logger.info(
                    "[ID CHECK]   raw_point_id=%s  payload_chunk_id=%s",
                    raw_point_id,
                    payload_chunk_id,
                )
 
            logger.info(
                "[ID CHECK] chunk_store sample IDs : %s",
                self.chunk_ids[:3],
            )
 
            # Check if any of the 3 Qdrant payload IDs exist in chunk_store
            qdrant_ids = {
                pt.payload.get("chunk_id", "")
                for pt in sample_points
            }
            store_id_set = set(self.chunk_ids[:500])  # check first 500 for speed
            overlap = qdrant_ids & store_id_set
            if overlap:
                logger.info(
                    "[ID CHECK] ✓ IDs match — found %d overlap in first 500 "
                    "chunk_store entries. Hybrid merge should work.",
                    len(overlap),
                )
            else:
                logger.warning(
                    "[ID CHECK] ✗ NO overlap found between Qdrant payload "
                    "chunk_ids and chunk_store IDs!\n"
                    "           This will cause both=0 in every merge.\n"
                    "           Compare the ID formats above and fix index.py "
                    "or the chunk_id field in preprocess_and_chunk.py."
                )
        except Exception as exc:
            logger.warning("[ID CHECK] Could not run ID check: %s", exc)
        logger.info("=" * 60)
 
    # ──────────────────────────────────────────
    # [FIX 1] Query encoding — mean pool to match embed_chunks.py
    # ──────────────────────────────────────────
    @torch.no_grad()
    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode query with "query: " prefix and MEAN POOLING.

        Mirrors embed_chunks.py:embed_batch() exactly:
        1. "query: " prefix (asymmetric encoding for arctic-embed)
        2. Tokenise with padding + truncation at max_length=512
        3. Forward pass — NO autocast (model already in float16)
        4. Mean-pool last_hidden_state weighted by attention_mask
        5. L2-normalise
        6. Return float32 numpy array shape (768,)

        NOTE: Do NOT wrap in torch.autocast here. embed_chunks.py uses
        autocast because it loads the model in default float32 and casts
        on-the-fly. Here the model is already loaded as float16 via
        dtype=torch.float16 in from_pretrained(). Double-casting degrades
        numerical precision and collapses cosine similarity scores.
        """
        prefixed = "query: " + query

        encoded = self.embed_tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # No autocast — model is already float16
        outputs = self.embed_model(**encoded)

        # Mean pooling — identical to embed_chunks.py:mean_pool()
        hidden   = outputs.last_hidden_state.float()        # (1, seq_len, 768)
        mask     = encoded["attention_mask"]                # (1, seq_len)
        mask_exp = mask.unsqueeze(-1).expand(hidden.size()).float()
        summed   = torch.sum(hidden * mask_exp, dim=1)
        counts   = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
        pooled   = summed / counts                          # (1, 768)
        embedding = F.normalize(pooled, p=2, dim=1)

        return embedding.cpu().float().numpy()[0]           # shape: (768,)
 
    # ──────────────────────────────────────────
    # 3. Dense retrieval (Qdrant)
    # ──────────────────────────────────────────
    def _dense_search(
        self,
        query_vector: np.ndarray,
        top_k: int = DENSE_TOP_K,
    ) -> list[RetrievedChunk]:
        """
        Search Qdrant HNSW index for semantically similar chunks.
        Uses query_points() API (qdrant-client >= 1.10).
 
        With the corrected mean-pool encoder, expect top scores in the
        0.65–0.90 range for relevant legal text (not 0.28 as before).
        """
        response = self.qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        results = response.points
 
        if results:
            logger.info(
                "Dense search: %d results  |  top score=%.4f  |  "
                "bottom score=%.4f",
                len(results),
                results[0].score,
                results[-1].score,
            )
        else:
            logger.warning("Dense search returned 0 results — check embedding model")
 
        chunks = []
        for r in results:
            p = r.payload or {}
            chunks.append(RetrievedChunk(
                chunk_id      = p.get("chunk_id", str(r.id)),
                text          = p.get("text", ""),
                case_title    = p.get("case_title", "Unknown"),
                year          = p.get("year", 0),
                petitioner    = p.get("petitioner", ""),
                respondent    = p.get("respondent", ""),
                judgment_date = p.get("judgment_date", ""),
                chunk_index   = p.get("chunk_index", 0),
                dense_score   = float(r.score),
                source        = "dense",
            ))
 
        return chunks
 
    # ──────────────────────────────────────────
    # 4. Sparse retrieval (BM25)
    # ──────────────────────────────────────────
    def _sparse_search(
        self,
        query: str,
        top_k: int = SPARSE_TOP_K,
    ) -> list[RetrievedChunk]:
        """
        Search BM25 index for keyword/citation matches.
        BM25 excels at exact terms: "AIR 1978 SC 597", "Section 438 CrPC".
        Dense embeddings miss these — BM25 is essential alongside dense.
        """
        tokenized_query = query.lower().split()
        scores          = self.bm25.get_scores(tokenized_query)
        top_indices     = np.argsort(scores)[::-1][:top_k]
 
        chunks = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            meta = self.metadatas[idx]
            chunks.append(RetrievedChunk(
                chunk_id      = self.chunk_ids[idx],
                text          = self.texts[idx],
                case_title    = meta.get("case_title", "Unknown"),
                year          = meta.get("year", 0),
                petitioner    = meta.get("petitioner", ""),
                respondent    = meta.get("respondent", ""),
                judgment_date = meta.get("judgment_date", ""),
                chunk_index   = meta.get("chunk_index", 0),
                sparse_score  = float(scores[idx]),
                source        = "sparse",
            ))
 
        return chunks
 
    # ──────────────────────────────────────────
    # [FIX 2] Merge with Reciprocal Rank Fusion
    # ──────────────────────────────────────────
    def _merge_results(
        self,
        dense_chunks  : list[RetrievedChunk],
        sparse_chunks : list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Merge dense + sparse using Reciprocal Rank Fusion (RRF).
 
        WHY RRF instead of raw score addition:
          Dense scores:  0.23 – 0.28  (cosine similarity, unit-bounded)
          Sparse scores: 29.0 – 34.6  (BM25 Okapi, unbounded)
          Direct addition → sparse always dominates → dense ignored → both=0.
 
        RRF formula (Cormack et al., 2009):
          rrf(d) = Σ  1 / (k + rank_i(d))
          where rank_i(d) is the rank of document d in list i,
          and k=60 is a smoothing constant.
 
        RRF uses RANKS not raw scores, so scale differences between
        dense and sparse are completely eliminated. A document ranked
        #1 by dense gets 1/(60+1) ≈ 0.0164 regardless of its 0.28 score,
        and a document ranked #1 by sparse also gets 1/(60+1) ≈ 0.0164.
        When the same document appears in BOTH lists its scores add up,
        making genuinely relevant documents rise to the top.
 
        After RRF fusion the merged list is sorted by rrf_score and passed
        to the cross-encoder reranker, which performs the final fine-grained
        relevance scoring.
        """
        merged: dict[str, RetrievedChunk] = {}
 
        # ── Assign RRF contributions from dense results ────────────────
        for rank, chunk in enumerate(dense_chunks, start=1):
            chunk.rrf_score = 1.0 / (RRF_K + rank)
            merged[chunk.chunk_id] = chunk
 
        # ── Merge sparse — add RRF contribution, update source ─────────
        for rank, chunk in enumerate(sparse_chunks, start=1):
            rrf_contrib = 1.0 / (RRF_K + rank)
            if chunk.chunk_id in merged:
                # Chunk appears in BOTH — accumulate RRF and record scores
                existing               = merged[chunk.chunk_id]
                existing.sparse_score  = chunk.sparse_score
                existing.rrf_score    += rrf_contrib
                existing.source        = "both"
            else:
                # Sparse-only chunk
                chunk.rrf_score = rrf_contrib
                merged[chunk.chunk_id] = chunk
 
        # Sort by RRF score descending before sending to reranker
        result = sorted(
            merged.values(),
            key=lambda c: c.rrf_score,
            reverse=True,
        )
 
        # ── Diagnostic logging ─────────────────────────────────────────
        both_count  = sum(1 for c in result if c.source == "both")
        dense_only  = sum(1 for c in result if c.source == "dense")
        sparse_only = sum(1 for c in result if c.source == "sparse")
        logger.info(
            "Merge (RRF k=%d): both=%d  dense_only=%d  sparse_only=%d  total=%d",
            RRF_K, both_count, dense_only, sparse_only, len(result),
        )
 
        # Show top RRF results so we can confirm both signals contribute
        for c in result[:3]:
            logger.info(
                "  Top RRF: %s  rrf=%.4f  dense=%.4f  sparse=%.4f  source=%s",
                c.chunk_id[:60], c.rrf_score, c.dense_score,
                c.sparse_score, c.source,
            )
 
        # Warn if IDs still don't match after fix
        if both_count == 0:
            logger.warning(
                "still both=0 after RRF — chunk_id format mismatch between "
                "Qdrant payload and chunk_store. See [ID CHECK] output above."
            )
 
        return result
 
    # ──────────────────────────────────────────
    # 5. Reranking
    # ──────────────────────────────────────────
    def _rerank(
        self,
        query       : str,
        candidates  : list[RetrievedChunk],
        top_k       : int = RERANK_TOP_K,
    ) -> list[RetrievedChunk]:
        """
        Rerank candidates using bge-reranker-v2-m3.
 
        CrossEncoder reads (query, document) together — sees token
        interactions → much more accurate relevance than cosine similarity.
        We batch in groups of 32 to fit comfortably on A4000.
        """
        if not candidates:
            return []
 
        all_scores = []
        batch_size = 32
 
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            pairs = [(query, c.text) for c in batch]
 
            encoded = self.reranker_tokenizer(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
 
            with torch.no_grad():
                outputs      = self.reranker_model(**encoded)
                batch_scores = outputs.logits.squeeze(-1).float().cpu().tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
            all_scores.extend(batch_scores)
 
        for chunk, score in zip(candidates, all_scores):
            chunk.rerank_score = float(score)
 
        reranked = sorted(
            candidates,
            key=lambda c: c.rerank_score,
            reverse=True,
        )[:top_k]
 
        logger.debug(
            "Reranked %d -> %d  |  top score: %.4f",
            len(candidates), len(reranked),
            reranked[0].rerank_score if reranked else 0,
        )
        return reranked
 
    # ──────────────────────────────────────────
    # 5b. Citation detection + boosting
    # ──────────────────────────────────────────
    def _is_citation_query(self, query: str) -> bool:
        """Detect if query is a pure legal citation."""
        import re
        patterns = [
            r"AIR\s+\d{4}\s+SC\s+\d+",          # AIR 1978 SC 597
            r"\(\d{4}\)\s+\d+\s+SCC\s+\d+",      # (1978) 2 SCC 597
            r"\d{4}\s+SCR\s+\(\d+\)\s+\d+",      # 1978 SCR (2) 621
        ]
        return any(re.search(p, query, re.IGNORECASE) for p in patterns)
 
    def _boost_citation_match(
        self,
        query  : str,
        chunks : list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        If query is a citation, boost chunks whose text contains the
        exact citation string to rank #1.
        Strips spaces before comparing to handle "AIR 1978 SC 597"
        matching "AIR1978SC597" in text.
        """
        citation = query.strip()
        for chunk in chunks:
            if citation.replace(" ", "") in chunk.text.replace(" ", ""):
                chunk.rerank_score += 10.0   # strong boost — floats to top
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
 
    # ──────────────────────────────────────────
    # 6. Build LLM prompt
    # ──────────────────────────────────────────
    def _build_prompt(
        self,
        query   : str,
        chunks  : list[RetrievedChunk],
    ) -> str:
        """
        Build structured prompt matching the fine-tuning format exactly.
        Format must match format_dataset.py → format_to_instruction.
        """
        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            block = (
                f"[{i}] Case: {chunk.case_title}\n"
                f"     Date: {chunk.judgment_date or str(chunk.year)}\n"
                f"     Excerpt: {chunk.text}"
            )
            context_blocks.append(block)
 
        context = "\n\n".join(context_blocks)
 
        prompt = (
            f"<s>[INST] <<SYS>>\n"
            f"{SYSTEM_PROMPT}\n"
            f"<</SYS>>\n\n"
            f"The following are excerpts from Supreme Court of India "
            f"judgments relevant to the question:\n\n"
            f"{context}\n\n"
            f"Based on the above judgment excerpts, answer this question:\n"
            f"Question: {query} [/INST]\n\n"
        )
 
        return prompt
 
    # ──────────────────────────────────────────
    # 7a. LLM generation — Ollama
    # ──────────────────────────────────────────
    def _generate_ollama(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model"  : self.ollama_model,
                    "prompt" : prompt,
                    "stream" : False,
                    "options": {
                        "temperature" : LLM_TEMPERATURE,
                        "num_predict" : LLM_MAX_NEW_TOKENS,
                        "stop"        : ["</s>", "[INST]", "Question:"],
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
 
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Start it with:\n"
                "  ollama serve\n"
                "Or open the Ollama application."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                "Ollama timed out. The model may be loading. "
                "Try again in 30 seconds."
            )
 
    # ──────────────────────────────────────────
    # 7b. LLM generation — HuggingFace
    # ──────────────────────────────────────────
    @torch.no_grad()
    def _generate_hf(self, prompt: str) -> str:
        if self.saul_model is None or self.saul_tokenizer is None:
            raise RuntimeError(
                "HuggingFace backend not loaded. "
                "Initialize with llm_backend='hf'."
            )
 
        inputs = self.saul_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).to(self.device)
 
        outputs = self.saul_model.generate(
            **inputs,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
            do_sample=True,
            pad_token_id=self.saul_tokenizer.pad_token_id,
            eos_token_id=self.saul_tokenizer.eos_token_id,
        )
 
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        answer    = self.saul_tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()
 
        return answer
 
    # ──────────────────────────────────────────
    # 8. Main retrieve() function
    # ──────────────────────────────────────────
    def retrieve(self, query: str) -> QueryResult:
        logger.info("Query: %s", query)
        t_total = time.time()
        latency = {}
 
        t0 = time.time()
        query_vector = self._encode_query(query)
        latency["encode_ms"] = round((time.time() - t0) * 1000, 1)
 
        t0 = time.time()
        dense_chunks = self._dense_search(query_vector)
        latency["dense_ms"] = round((time.time() - t0) * 1000, 1)
 
        t0 = time.time()
        sparse_chunks = self._sparse_search(query)
        latency["sparse_ms"] = round((time.time() - t0) * 1000, 1)
 
        t0 = time.time()
        candidates = self._merge_results(dense_chunks, sparse_chunks)
        latency["merge_ms"] = round((time.time() - t0) * 1000, 1)
 
        t0 = time.time()
        top_chunks = self._rerank(query, candidates)
        latency["rerank_ms"] = round((time.time() - t0) * 1000, 1)
 
        if self._is_citation_query(query):
            top_chunks = self._boost_citation_match(query, top_chunks)
            logger.info(
                "Citation query detected — boosting exact matches: %s", query
            )
 
        answer = ""
        latency["llm_ms"] = 0
 
        if self.llm_backend != "none" and top_chunks:
            t0     = time.time()
            prompt = self._build_prompt(query, top_chunks)
 
            if self.llm_backend == "ollama":
                answer = self._generate_ollama(prompt)
            elif self.llm_backend == "hf":
                answer = self._generate_hf(prompt)
 
            latency["llm_ms"] = round((time.time() - t0) * 1000, 1)
 
        total_ms = round((time.time() - t_total) * 1000, 1)
 
        sources = [
            {
                "rank"          : i + 1,
                "chunk_id"      : c.chunk_id,
                "case_title"    : c.case_title,
                "year"          : c.year,
                "judgment_date" : c.judgment_date,
                "petitioner"    : c.petitioner,
                "respondent"    : c.respondent,
                "text_preview"  : c.text[:200] + "..." if len(c.text) > 200 else c.text,
                "rerank_score"  : round(c.rerank_score, 4),
                "dense_score"   : round(c.dense_score, 4),
                "sparse_score"  : round(c.sparse_score, 4),
                "rrf_score"     : round(c.rrf_score, 4),
                "source"        : c.source,
            }
            for i, c in enumerate(top_chunks)
        ]
 
        result = QueryResult(
            query       = query,
            answer      = answer,
            sources     = sources,
            latency_ms  = latency,
            total_ms    = total_ms,
            llm_backend = self.llm_backend,
        )
 
        self._log_query(result)
 
        logger.info(
            "Latency | encode=%.0fms dense=%.0fms sparse=%.0fms "
            "rerank=%.0fms llm=%.0fms | total=%.0fms",
            latency["encode_ms"],
            latency["dense_ms"],
            latency["sparse_ms"],
            latency["rerank_ms"],
            latency["llm_ms"],
            total_ms,
        )
 
        return result
 
    # ──────────────────────────────────────────
    # 9. Query logging
    # ──────────────────────────────────────────
    def _log_query(self, result: QueryResult) -> None:
        log_entry = {
            "timestamp"  : time.strftime("%Y-%m-%dT%H:%M:%S"),
            "query"      : result.query,
            "answer"     : result.answer,
            "sources"    : result.sources,
            "latency_ms" : result.latency_ms,
            "total_ms"   : result.total_ms,
            "llm_backend": result.llm_backend,
        }
 
        with open(QUERY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
 
 
# ─────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────
def display_result(result: QueryResult) -> None:
    print("\n" + "═" * 70)
    print(f"  QUERY: {result.query}")
    print("═" * 70)
 
    if result.answer:
        print("\n  ANSWER:")
        print("  " + result.answer.replace("\n", "\n  "))
    else:
        print("\n  [Retrieval-only mode — no LLM answer generated]")
 
    print("\n" + "─" * 70)
    print("  SOURCES (top chunks used as context):")
    print("─" * 70)
 
    for src in result.sources:
        print(f"\n  [{src['rank']}] {src['case_title']}  ({src['year']})")
        print(f"      Date    : {src['judgment_date']}")
        print(
            f"      Scores  : rerank={src['rerank_score']:.4f}  "
            f"dense={src['dense_score']:.4f}  "
            f"sparse={src['sparse_score']:.4f}  "
            f"rrf={src['rrf_score']:.4f}  "
            f"[{src['source']}]"
        )
        print(f"      Preview : {src['text_preview']}")
 
    print("\n" + "─" * 70)
    print(
        f"  LATENCY: encode={result.latency_ms['encode_ms']:.0f}ms  "
        f"dense={result.latency_ms['dense_ms']:.0f}ms  "
        f"sparse={result.latency_ms['sparse_ms']:.0f}ms  "
        f"rerank={result.latency_ms['rerank_ms']:.0f}ms  "
        f"llm={result.latency_ms['llm_ms']:.0f}ms  "
        f"| total={result.total_ms:.0f}ms"
    )
    print("═" * 70 + "\n")
 
 
# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 4: Retrieve and answer legal questions "
            "using the Supreme Court of India judgment corpus."
        )
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Single query. If not provided, runs in interactive mode.",
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        choices=["ollama", "hf", "none"],
        default="ollama",
        help="LLM backend: ollama | hf | none",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name (default: {DEFAULT_OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default=QDRANT_HOST,
        help=f"Qdrant host (default: {QDRANT_HOST})",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=QDRANT_PORT,
        help=f"Qdrant port (default: {QDRANT_PORT})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RERANK_TOP_K,
        help=f"Final chunks to pass to LLM (default: {RERANK_TOP_K})",
    )
    return parser.parse_args()
 
 
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main() -> None:
    args = parse_args()
 
    logger.info("=" * 60)
    logger.info("retrieve.py — Stage 4: Retrieval + Answer Generation")
    logger.info("LLM backend : %s", args.llm_backend)
    if args.llm_backend == "ollama":
        logger.info("Ollama model: %s", args.ollama_model)
    logger.info("Qdrant      : %s:%d", args.qdrant_host, args.qdrant_port)
    logger.info("=" * 60)
 
    retriever = LegalRetriever(
        llm_backend  = args.llm_backend,
        ollama_model = args.ollama_model,
        qdrant_host  = args.qdrant_host,
        qdrant_port  = args.qdrant_port,
    )
 
    if args.query:
        result = retriever.retrieve(args.query)
        display_result(result)
        return
 
    print("\n" + "═" * 70)
    print("  Supreme Court of India — Legal Research Assistant")
    print("  Powered by: arctic-embed-m + Qdrant + BM25 + SaulLM-7B")
    print("  Corpus: 919,518 chunks | 75 years (1950-2025)")
    print("═" * 70)
    print("  Type your legal question and press Enter.")
    print("  Examples:")
    print("    → What is the right to life under Article 21?")
    print("    → AIR 1978 SC 597 Maneka Gandhi")
    print("    → Can anticipatory bail be cancelled after grant?")
    print("    → What is the test for granting interim injunction?")
    print("  Type 'quit' or 'exit' to stop.")
    print("═" * 70 + "\n")
 
    while True:
        try:
            query = input("  Your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break
 
        if not query:
            continue
 
        if query.lower() in ("quit", "exit", "q"):
            print("  Goodbye.")
            break
 
        try:
            result = retriever.retrieve(query)
            display_result(result)
        except RuntimeError as e:
            print(f"\n  ERROR: {e}\n")
        except Exception as e:
            logger.error("Unexpected error: %s", e, exc_info=True)
            print(f"\n  Unexpected error: {e}\n")
 
 
if __name__ == "__main__":
    # Better CUDA memory behavior for long runs and large corpora.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()