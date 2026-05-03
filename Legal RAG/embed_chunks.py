"""
Embed chunked Supreme Court text using a Hugging Face model on GPU.

Default model:
    Snowflake/snowflake-arctic-embed-m-v2.0

Outputs (inside --output-dir):
    - embeddings.npy: float32 matrix of shape (num_chunks, embedding_dim)
    - embedding_manifest.json: metadata about the embedding run

Example:
    python embed_chunks.py \
      --input-jsonl chunks.jsonl \
      --output-dir embeddings_arctic_m_v2 \
      --batch-size 64 \
      --max-length 512
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Iterable

if os.name == "nt":
    # Workaround for duplicate OpenMP runtime initialization on some Windows envs.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from huggingface_hub import snapshot_download
from numpy.lib.format import open_memmap
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

DEFAULT_MODEL = "Snowflake/snowflake-arctic-embed-m-v2.0"
DEFAULT_INPUT = Path(__file__).resolve().parent / "chunks.jsonl"


def model_name_to_local_dirname(model_name: str) -> str:
    return model_name.replace("/", "--")


def download_model_to_local_dir(model_name: str, local_model_dir: Path) -> None:
    """Download a full local snapshot without symlinked cache indirection."""
    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_model_dir),
        force_download=True,
        # Speed up and reduce disk usage; these files are not used for PyTorch inference.
        ignore_patterns=["onnx/*"],
    )


def load_model_and_tokenizer(model_name: str, local_model_dir: Path) -> tuple[AutoTokenizer, AutoModel]:
    """
    Load tokenizer/model and repair a broken local Hugging Face cache if needed.

    Some interrupted downloads leave an incomplete snapshot directory in
    ~/.cache/huggingface/hub, which causes FileNotFoundError for files that
    do exist in the remote model repo.
    """
    base_kwargs = {"trust_remote_code": True, "local_files_only": True}
    local_model_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the local directory has a complete copy of the model files.
    if not (local_model_dir / "tokenizer_config.json").exists():
        print(f"Preparing local model files in: {local_model_dir}")
        download_model_to_local_dir(model_name, local_model_dir)

    def load_config_with_safe_attention() -> AutoConfig:
        config = AutoConfig.from_pretrained(str(local_model_dir), **base_kwargs)
        # This model can request xformers-only attention; disable it for wider compatibility.
        if hasattr(config, "use_memory_efficient_attention"):
            setattr(config, "use_memory_efficient_attention", False)
        if hasattr(config, "memory_efficient_attention"):
            setattr(config, "memory_efficient_attention", False)
        if hasattr(config, "attn_implementation"):
            setattr(config, "attn_implementation", "eager")
        return config

    try:
        config = load_config_with_safe_attention()
        tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir), **base_kwargs)
        model = AutoModel.from_pretrained(str(local_model_dir), config=config, **base_kwargs)
        return tokenizer, model
    except (FileNotFoundError, OSError) as exc:
        print("Detected incomplete local model directory. Re-downloading...")
        print(f"Original error: {exc}")

        shutil.rmtree(local_model_dir, ignore_errors=True)
        local_model_dir.mkdir(parents=True, exist_ok=True)
        download_model_to_local_dir(model_name, local_model_dir)

        config = load_config_with_safe_attention()
        tokenizer = AutoTokenizer.from_pretrained(
            str(local_model_dir),
            **base_kwargs,
        )
        model = AutoModel.from_pretrained(
            str(local_model_dir),
            config=config,
            **base_kwargs,
        )
        return tokenizer, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GPU embeddings from chunks.jsonl using Hugging Face models."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to chunks.jsonl (default: ./chunks.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "embeddings_arctic_m_v2",
        help="Directory to save embeddings.npy and embedding_manifest.json",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face model id for embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation (reduce if CUDA OOM)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max token length for truncation",
    )
    parser.add_argument(
        "--doc-prefix",
        type=str,
        default="",
        help="Optional prefix added before each chunk text (e.g., 'passage: ')",
    )
    parser.add_argument(
        "--disable-fp16",
        action="store_true",
        help="Disable FP16 autocast on CUDA",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback if CUDA is unavailable",
    )
    return parser.parse_args()


def count_lines(file_path: Path) -> int:
    with file_path.open("rb") as f:
        return sum(1 for _ in f)


def iter_chunk_texts(input_jsonl: Path, doc_prefix: str) -> Iterable[str]:
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                raise ValueError(f"Empty line found at {line_no} in {input_jsonl}")
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            text = obj.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing/empty 'text' at line {line_no} in {input_jsonl}")
            yield f"{doc_prefix}{text}"


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_batch(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int,
    use_fp16: bool,
) -> np.ndarray:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.inference_mode():
        if device.type == "cuda" and use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**encoded)
        else:
            outputs = model(**encoded)

        pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

    return pooled.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()

    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_jsonl}")

    use_cuda = torch.cuda.is_available()
    if not use_cuda and not args.allow_cpu:
        raise RuntimeError(
            "CUDA GPU not found. This script defaults to GPU mode. "
            "Use --allow-cpu only if you intentionally want CPU embeddings."
        )

    device = torch.device("cuda" if use_cuda else "cpu")
    use_fp16 = bool(use_cuda and not args.disable_fp16)

    print(f"Loading model: {args.model_name}")
    print(f"Device: {device}")
    print(f"FP16 autocast: {use_fp16}")

    local_models_root = Path(__file__).resolve().parent / ".hf_models"
    local_model_dir = local_models_root / model_name_to_local_dirname(args.model_name)
    tokenizer, model = load_model_and_tokenizer(args.model_name, local_model_dir)
    model.to(device)
    model.eval()

    total_rows = count_lines(args.input_jsonl)
    if total_rows == 0:
        raise ValueError(f"Input file has 0 lines: {args.input_jsonl}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = args.output_dir / "embeddings.npy"
    manifest_path = args.output_dir / "embedding_manifest.json"

    start = time.time()
    mmap = None
    write_idx = 0
    batch: list[str] = []

    progress = tqdm(total=total_rows, desc="Embedding chunks", unit="chunk")

    for text in iter_chunk_texts(args.input_jsonl, args.doc_prefix):
        batch.append(text)
        if len(batch) < args.batch_size:
            continue

        emb = embed_batch(batch, tokenizer, model, device, args.max_length, use_fp16)
        if mmap is None:
            mmap = open_memmap(
                emb_path,
                mode="w+",
                dtype=np.float32,
                shape=(total_rows, emb.shape[1]),
            )

        mmap[write_idx : write_idx + emb.shape[0]] = emb
        write_idx += emb.shape[0]
        progress.update(len(batch))
        batch = []

    if batch:
        emb = embed_batch(batch, tokenizer, model, device, args.max_length, use_fp16)
        if mmap is None:
            mmap = open_memmap(
                emb_path,
                mode="w+",
                dtype=np.float32,
                shape=(total_rows, emb.shape[1]),
            )
        mmap[write_idx : write_idx + emb.shape[0]] = emb
        write_idx += emb.shape[0]
        progress.update(len(batch))

    progress.close()

    if mmap is None or write_idx != total_rows:
        raise RuntimeError(
            f"Embedding row mismatch: expected {total_rows}, wrote {write_idx}"
        )

    manifest = {
        "model_name": args.model_name,
        "input_jsonl": str(args.input_jsonl.resolve()),
        "embeddings_path": str(emb_path.resolve()),
        "num_chunks": total_rows,
        "embedding_dim": int(mmap.shape[1]),
        "dtype": "float32",
        "normalized": True,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "doc_prefix": args.doc_prefix,
        "device": str(device),
        "fp16_autocast": use_fp16,
        "elapsed_seconds": round(time.time() - start, 2),
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Embedding complete.")
    print(f"Saved embeddings: {emb_path}")
    print(f"Saved manifest:  {manifest_path}")
    print(f"Shape: ({manifest['num_chunks']}, {manifest['embedding_dim']})")


if __name__ == "__main__":
    # Better CUDA memory behavior for long runs and large corpora.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
