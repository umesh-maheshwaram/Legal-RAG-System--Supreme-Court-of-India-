"""
finetune.py — QLoRA Fine-tuning of SaulLM-7B on IndicLegalQA
Supreme Court Judgments RAG Pipeline — Stage 4 LLM Preparation

Base model : .hf_models/Saul-7B-Base  (local — no internet download)
Dataset    : train_dataset.jsonl + val_dataset.jsonl
Method     : QLoRA — 4-bit quantization + LoRA adapters
"""

# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────
import os
import gc
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Windows: suppress duplicate OpenMP warning
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Disable tokenizer parallelism (causes deadlocks in DataLoader)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force fully offline mode — model is already local
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig
class DataCollatorForCompletionOnlyLM:
    """
    Masks prompt tokens so the model trains only on the answer portion.
    Handles padding of variable-length sequences within a batch.
    """
    def __init__(self, response_template_ids: list, tokenizer):
        self.response_template_ids = response_template_ids
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: list) -> dict:
        import torch

        # Pad all sequences to the length of the longest in the batch
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        tmpl = self.response_template_ids
        tlen = len(tmpl)

        for f in features:
            ids  = list(f["input_ids"])
            attn = list(f.get("attention_mask", [1] * len(ids)))

            # Pad to max_len
            pad_len = max_len - len(ids)
            ids_padded  = ids  + [self.pad_token_id] * pad_len
            attn_padded = attn + [0] * pad_len

            # Build labels: mask prompt tokens with -100
            labels = list(ids_padded)
            mask_until = 0
            for j in range(len(ids) - tlen + 1):
                if ids[j : j + tlen] == tmpl:
                    mask_until = j + tlen
            # Mask prompt + padding
            for j in range(mask_until):
                labels[j] = -100
            for j in range(len(ids), max_len):  # mask padding positions
                labels[j] = -100

            input_ids_batch.append(ids_padded)
            attention_mask_batch.append(attn_padded)
            labels_batch.append(labels)

        return {
            "input_ids":      torch.tensor(input_ids_batch,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels":         torch.tensor(labels_batch,         dtype=torch.long),
        }


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / ".hf_models" / "Saul-7B-Base"
TRAIN_FILE = ROOT_DIR / "train_dataset.jsonl"
VAL_FILE   = ROOT_DIR / "val_dataset.jsonl"
OUTPUT_DIR = ROOT_DIR / "saullm_indic_legal"
MERGED_DIR = ROOT_DIR / "saullm_indic_legal_merged"
LOG_FILE   = ROOT_DIR / "finetune.log"


# ─────────────────────────────────────────────
# Startup check
# ─────────────────────────────────────────────
def _check_model_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nLocal model folder not found: {path}\n"
            f"Make sure your downloaded model is at:\n  {path}"
        )
    required = ["config.json", "tokenizer_config.json"]
    missing  = [f for f in required if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"\nModel folder found but missing files: {missing}"
        )
    safetensors = list(path.glob("*.safetensors"))
    bin_files   = list(path.glob("pytorch_model*.bin"))
    if not safetensors and not bin_files:
        raise FileNotFoundError(
            f"\nNo model weight files (*.safetensors or *.bin) found in {path}"
        )


# ─────────────────────────────────────────────
# Model & Training Configuration
# ─────────────────────────────────────────────

BNBCONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# ── Hyperparameters ───────────────────────────
# warmup_ratio removed (deprecated in newer transformers — use warmup_steps)
# group_by_length removed (deprecated in newer transformers)
# tokenizer removed from SFTTrainer (deprecated in newer trl — use processing_class)
# dataset_text_field and max_seq_length moved to SFTConfig (not SFTTrainer kwargs)
TRAIN_CONFIG = dict(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,                # replaces warmup_ratio (deprecated in v5.2)
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    dataloader_num_workers=0,
    seed=42,
    data_seed=42,
)

MAX_SEQ_LENGTH    = 512
RESPONSE_TEMPLATE = "[/INST]"


def _compute_dtype() -> torch.dtype:
    return torch.bfloat16 if TRAIN_CONFIG.get("bf16", False) else torch.float16


def configure_precision() -> None:
    """
    Configure mixed precision and 4-bit compute dtype to avoid AMP scaler
    mismatches (e.g., fp16 scaler trying to unscale bf16 gradients).
    """
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if use_bf16:
        TRAIN_CONFIG["bf16"] = True
        TRAIN_CONFIG["fp16"] = False
        BNBCONFIG.bnb_4bit_compute_dtype = torch.bfloat16
        logger.info("Precision mode: bf16 (AMP scaler disabled)")
    else:
        TRAIN_CONFIG["bf16"] = False
        TRAIN_CONFIG["fp16"] = True
        BNBCONFIG.bnb_4bit_compute_dtype = torch.float16
        logger.info("Precision mode: fp16")

    logger.info("4-bit compute dtype: %s", BNBCONFIG.bnb_4bit_compute_dtype)


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
# 1. Validate inputs
# ─────────────────────────────────────────────
def validate_inputs(train_file: Path, val_file: Path) -> None:
    for path, expected_min, label in [
        (train_file, 7000, "train"),
        (val_file,   1000, "val"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} file not found: {path}\n"
                "Run format_dataset.py first."
            )
        with open(path, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        if count < expected_min:
            raise ValueError(
                f"{label} file has only {count} records "
                f"(expected at least {expected_min})."
            )
        logger.info("%s file: %d records  (%s)", label.capitalize(), count, path)


# ─────────────────────────────────────────────
# 2. Load datasets
# ─────────────────────────────────────────────
def load_datasets(
    train_file: Path,
    val_file: Path,
    dry_run: bool = False,
) -> DatasetDict:
    logger.info("Loading datasets...")
    dataset = load_dataset(
        "json",
        data_files={"train": str(train_file), "validation": str(val_file)},
    )
    if dry_run:
        logger.info("DRY RUN — truncating to 200 train + 50 val records")
        dataset["train"]      = dataset["train"].select(range(200))
        dataset["validation"] = dataset["validation"].select(range(50))
    logger.info(
        "Dataset loaded  |  train=%d  |  val=%d",
        len(dataset["train"]), len(dataset["validation"]),
    )
    return dataset


# ─────────────────────────────────────────────
# 3. Load tokenizer
# ─────────────────────────────────────────────
def load_tokenizer(model_path: Path) -> AutoTokenizer:
    logger.info("Loading tokenizer from local path: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        padding_side="right",
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token = eos_token (%s)", tokenizer.eos_token)
    logger.info(
        "Tokenizer loaded  |  vocab_size=%d  |  pad='%s'",
        tokenizer.vocab_size, tokenizer.pad_token,
    )
    return tokenizer


# ─────────────────────────────────────────────
# 4. Load model in 4-bit (QLoRA)
# ─────────────────────────────────────────────
def load_model(model_path: Path) -> AutoModelForCausalLM:
    logger.info("Loading model from local path: %s", model_path)
    logger.info("VRAM before load: %.1f GB used", _vram_used_gb())

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=BNBCONFIG,
        device_map={"": 0},
        trust_remote_code=True,
        dtype=_compute_dtype(),
        local_files_only=True,
    )

    logger.info("VRAM after model load: %.1f GB used", _vram_used_gb())
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    logger.info(
        "Model loaded  |  params=%.1fB  |  dtype=%s",
        sum(p.numel() for p in model.parameters()) / 1e9,
        next(model.parameters()).dtype,
    )
    return model


# ─────────────────────────────────────────────
# 5. Apply LoRA adapters
# ─────────────────────────────────────────────
def apply_lora(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    logger.info("Applying LoRA adapters  |  rank=%d  |  alpha=%d",
                LORA_CONFIG.r, LORA_CONFIG.lora_alpha)
    model     = get_peft_model(model, LORA_CONFIG)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    pct       = 100 * trainable / total
    logger.info(
        "LoRA applied  |  trainable=%s  |  total=%s  |  %.4f%%",
        _fmt_params(trainable), _fmt_params(total), pct,
    )
    if pct > 5.0:
        logger.warning(
            "Trainable params > 5%% — LoRA may not be applied correctly. "
            "Expected ~0.1%% for rank=16 on 7B model."
        )
    return model


# ─────────────────────────────────────────────
# 6. Build trainer
# ─────────────────────────────────────────────
def build_trainer(
    model      : AutoModelForCausalLM,
    tokenizer  : AutoTokenizer,
    dataset    : DatasetDict,
    output_dir : Path,
    dry_run    : bool,
) -> SFTTrainer:

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        **TRAIN_CONFIG,
    )

    # Tokenize each example manually, truncating to MAX_SEQ_LENGTH
    def formatting_func(example):
        return example["text"]

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    tokenized_train = dataset["train"].map(tokenize, remove_columns=dataset["train"].column_names)
    tokenized_val   = dataset["validation"].map(tokenize, remove_columns=dataset["validation"].column_names)

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids=tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False),
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    logger.info(
        "Trainer built  |  steps_per_epoch=%d  |  total_steps=%d",
        len(trainer.get_train_dataloader()),
        trainer.args.max_steps if trainer.args.max_steps > 0
        else len(trainer.get_train_dataloader()) * TRAIN_CONFIG["num_train_epochs"],
    )
    return trainer


# ─────────────────────────────────────────────
# 7. Run training
# ─────────────────────────────────────────────
def run_training(
    trainer: SFTTrainer,
    output_dir: Path,
    tokenizer: AutoTokenizer,
    dry_run: bool = False,
) -> dict:
    logger.info("=" * 60)
    logger.info("Starting QLoRA fine-tuning")
    logger.info("Output dir   : %s", output_dir)
    logger.info("Epochs       : %d", TRAIN_CONFIG["num_train_epochs"])
    logger.info("Batch size   : %d (effective: %d)",
                TRAIN_CONFIG["per_device_train_batch_size"],
                TRAIN_CONFIG["per_device_train_batch_size"] *
                TRAIN_CONFIG["gradient_accumulation_steps"])
    logger.info("Learning rate: %s", TRAIN_CONFIG["learning_rate"])
    logger.info("VRAM before training: %.1f GB used", _vram_used_gb())
    logger.info("=" * 60)

    import time
    t_start = time.time()

    resume_from = None
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    if checkpoints:
        latest_checkpoint = str(checkpoints[-1])
        if dry_run:
            logger.info("Dry run: skipping checkpoint resume.")
        elif _torch_version_lt_2_6():
            logger.warning(
                "Checkpoint resume disabled because torch %s is below 2.6 and "
                "transformers blocks torch.load for security reasons (CVE-2025-32434). "
                "Upgrade torch to >=2.6 to resume from optimizer state.",
                torch.__version__,
            )
        else:
            resume_from = latest_checkpoint
            logger.info("Resuming from checkpoint: %s", resume_from)

    train_result = trainer.train(resume_from_checkpoint=resume_from)
    elapsed      = time.time() - t_start

    logger.info(
        "Training complete  |  elapsed=%.1f min  |  steps=%d",
        elapsed / 60, train_result.global_step,
    )
    logger.info("VRAM after training: %.1f GB used", _vram_used_gb())

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("LoRA adapter saved → %s", output_dir)

    metrics  = train_result.metrics
    manifest = {
        "base_model"      : str(MODEL_PATH),
        "dataset_train"   : str(TRAIN_FILE),
        "dataset_val"     : str(VAL_FILE),
        "output_dir"      : str(output_dir),
        "lora_rank"       : LORA_CONFIG.r,
        "lora_alpha"      : LORA_CONFIG.lora_alpha,
        "lora_dropout"    : LORA_CONFIG.lora_dropout,
        "target_modules"  : sorted(list(LORA_CONFIG.target_modules)),
        "num_epochs"      : TRAIN_CONFIG["num_train_epochs"],
        "batch_size"      : TRAIN_CONFIG["per_device_train_batch_size"],
        "effective_batch" : (TRAIN_CONFIG["per_device_train_batch_size"] *
                             TRAIN_CONFIG["gradient_accumulation_steps"]),
        "learning_rate"   : TRAIN_CONFIG["learning_rate"],
        "max_seq_length"  : MAX_SEQ_LENGTH,
        "quantization"    : "4-bit NF4 (QLoRA)",
        "trained_at"      : datetime.now().isoformat(),
        "elapsed_minutes" : round(elapsed / 60, 1),
        "global_steps"    : train_result.global_step,
        "train_loss"      : round(metrics.get("train_loss", 0), 4),
    }
    manifest_path = output_dir / "training_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Training manifest saved → %s", manifest_path)
    return metrics


# ─────────────────────────────────────────────
# 8. Evaluate on validation set
# ─────────────────────────────────────────────
def run_evaluation(trainer: SFTTrainer) -> dict:
    logger.info("Running final evaluation on validation set...")
    eval_metrics = trainer.evaluate()
    logger.info("Evaluation results:")
    for key, val in eval_metrics.items():
        logger.info("  %-30s : %.4f", key, val)
    return eval_metrics


# ─────────────────────────────────────────────
# 9. Merge LoRA into base model (optional)
# ─────────────────────────────────────────────
def merge_and_save(output_dir: Path, merged_dir: Path) -> None:
    logger.info("Merging LoRA adapters into base model...")
    logger.info("VRAM before merge: %.1f GB used", _vram_used_gb())

    from peft import PeftModel

    def _load_and_merge(device_map):
        base_model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            dtype=_compute_dtype(),
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=True,
        )
        peft_model = PeftModel.from_pretrained(base_model, str(output_dir))
        merged = peft_model.merge_and_unload()
        return merged, peft_model, base_model

    try:
        target_device = {"": 0} if torch.cuda.is_available() else {"": "cpu"}
        logger.info("Merge load device_map: %s", target_device)
        merged_model, peft_model, base_model = _load_and_merge(target_device)
    except TypeError as e:
        if "unhashable type: 'set'" not in str(e):
            raise
        logger.warning("Auto mapping bug hit during merge; retrying on CPU.")
        merged_model, peft_model, base_model = _load_and_merge({"": "cpu"})

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(output_dir), trust_remote_code=True, local_files_only=True,
    )
    tokenizer.save_pretrained(str(merged_dir))

    size_gb = sum(f.stat().st_size for f in merged_dir.glob("*.safetensors")) / 1e9
    logger.info("Merged model saved → %s  (%.1f GB)", merged_dir, size_gb)

    del merged_model, peft_model, base_model
    gc.collect()
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# 10. Inference smoke test
# ─────────────────────────────────────────────
def run_inference_test(output_dir: Path, tokenizer: AutoTokenizer) -> None:
    logger.info("=" * 60)
    logger.info("Inference smoke test — 3 test queries")
    logger.info("=" * 60)

    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        quantization_config=BNBCONFIG,
        device_map={"": 0},
        trust_remote_code=True,
        dtype=_compute_dtype(),
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, str(output_dir))
    model.eval()

    test_queries = [
        {
            "case"    : "Maneka Gandhi vs Union Of India",
            "date"    : "25th January 1978",
            "question": "What fundamental right did the Supreme Court expand in this case?",
        },
        {
            "case"    : "Gurbaksh Singh Sibbia vs State of Punjab",
            "date"    : "9th April 1980",
            "question": "What did the Supreme Court rule regarding anticipatory bail?",
        },
        {
            "case"    : "State of Punjab vs Rakesh Kumar",
            "date"    : "3rd December 2018",
            "question": "What were the respondents convicted of?",
        },
    ]

    system_prompt = (
        "You are an expert legal assistant specializing in Supreme Court "
        "of India judgments. Answer the legal question accurately and "
        "concisely based on Indian Supreme Court jurisprudence. Always "
        "ground your answer in the specific case context provided. "
        "Cite the case name and judgment date when relevant."
    )

    for i, q in enumerate(test_queries, 1):
        prompt = (
            f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
            f"Case: {q['case']}\n"
            f"Date: {q['date']}\n"
            f"Question: {q['question']} [/INST]\n\n"
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH,
        ).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200, temperature=0.1, do_sample=True,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        answer    = tokenizer.decode(generated, skip_special_tokens=True).strip()
        logger.info("\n  [%d] Case : %s", i, q["case"])
        logger.info("      Q    : %s", q["question"])
        logger.info("      A    : %s", answer)

    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("=" * 60)
    logger.info("Smoke test complete")


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────
def _vram_used_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / 1e9
    return 0.0


def _fmt_params(n: int) -> str:
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.1f}M"
    return f"{n:,}"


def _torch_version_lt_2_6() -> bool:
    version_core = torch.__version__.split("+")[0]
    parts = version_core.split(".")
    try:
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return True
    return (major, minor) < (2, 6)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning of SaulLM-7B on IndicLegalQA (local model)."
    )
    parser.add_argument("--train-file",          type=Path, default=TRAIN_FILE)
    parser.add_argument("--val-file",            type=Path, default=VAL_FILE)
    parser.add_argument("--output-dir",          type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dry-run",             action="store_true",
        help="Train on 200 samples (~5 min) to verify pipeline before full run.")
    parser.add_argument("--merge",               action="store_true",
        help="After training, merge LoRA into base model for GGUF conversion.")
    parser.add_argument("--merge-only",          action="store_true",
        help="Skip training/eval and merge an existing LoRA adapter from --output-dir.")
    parser.add_argument("--skip-inference-test", action="store_true",
        help="Skip the inference smoke test after training.")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    set_seed(42)
    configure_precision()

    logger.info("=" * 60)
    logger.info("finetune.py — QLoRA fine-tuning  (LOCAL MODEL)")
    logger.info("Model path  : %s", MODEL_PATH)
    logger.info("Train file  : %s", args.train_file)
    logger.info("Val file    : %s", args.val_file)
    logger.info("Output dir  : %s", args.output_dir)
    logger.info("Dry run     : %s", args.dry_run)
    logger.info("Merge after : %s", args.merge)
    logger.info("Merge only  : %s", args.merge_only)
    if args.dry_run:
        logger.info("DRY RUN — 200 samples, ~5 minutes")
        logger.info("Run without --dry-run for full 4–6 hour training")
    logger.info("=" * 60)

    _check_model_path(MODEL_PATH)
    logger.info("Local model folder verified: %s", MODEL_PATH)

    if args.merge_only:
        if not args.output_dir.exists():
            raise FileNotFoundError(
                f"Adapter output dir not found: {args.output_dir}\n"
                "Run training first, or pass --output-dir to an existing LoRA adapter folder."
            )
        merge_and_save(args.output_dir, MERGED_DIR)
        logger.info("Merge-only complete")
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found. QLoRA fine-tuning requires a GPU.\n"
            "Verify with: python -c \"import torch; print(torch.cuda.is_available())\""
        )
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info("GPU  : %s", gpu_name)
    logger.info("VRAM : %.1f GB", vram_gb)
    if vram_gb < 14.0:
        raise RuntimeError(
            f"GPU has only {vram_gb:.1f} GB VRAM. "
            "QLoRA with SaulLM-7B requires at least 14 GB."
        )

    validate_inputs(args.train_file, args.val_file)
    dataset   = load_datasets(args.train_file, args.val_file, dry_run=args.dry_run)
    tokenizer = load_tokenizer(MODEL_PATH)
    model     = load_model(MODEL_PATH)
    model     = apply_lora(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(model, tokenizer, dataset, args.output_dir, args.dry_run)

    train_metrics = run_training(trainer, args.output_dir, tokenizer, dry_run=args.dry_run)
    eval_metrics  = run_evaluation(trainer)

    if not args.skip_inference_test:
        run_inference_test(args.output_dir, tokenizer)

    if args.merge:
        merge_and_save(args.output_dir, MERGED_DIR)

    logger.info("=" * 60)
    logger.info("Fine-tuning complete")
    logger.info("LoRA adapter : %s", args.output_dir)
    logger.info("Train loss   : %.4f", train_metrics.get("train_loss", 0))
    logger.info("Eval loss    : %.4f", eval_metrics.get("eval_loss", 0))
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Check eval_loss — target: 0.8–1.2 after 3 epochs")
    logger.info("  2. Run with --merge to create standalone model")
    logger.info("  3. Convert to GGUF for Ollama:")
    logger.info("       python convert_hf_to_gguf.py %s \\", MERGED_DIR)
    logger.info("         --outfile saullm_indic_legal_q4.gguf \\")
    logger.info("         --outtype q4_K_M")
    logger.info("  4. Load in Ollama:")
    logger.info("       ollama create saullm-legal -f Modelfile")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()