"""Shared paths and hyperparameters for bond-deal GLiNER2 two-stage training."""

from __future__ import annotations

import os
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TEST_ROOT.parent
WORKSPACE_ROOT = TEST_ROOT.parents[2]

# 原始数据路径：优先读环境变量 BOND_DATA，否则默认放在 test/data/ 下（打包时带上）。
RAW_DATA = Path(
    os.environ.get(
        "BOND_DATA",
        str(TEST_ROOT / "data" / "bond_deal_0805_structured_aug_v1_sample_10000.jsonl"),
    )
)
SCHEMA_DESC = TEST_ROOT / "schema" / "deal_field_descriptions.json"

DATA_DIR = TEST_ROOT / "data"
OUTPUT_DIR = TEST_ROOT / "outputs"
LOG_DIR = TEST_ROOT / "logs"

NER_DIR = OUTPUT_DIR / "ner"
STRUCTURE_DIR = OUTPUT_DIR / "structure"

# HuggingFace repo id; after ensure_model.py, training uses LOCAL_MODEL_DIR.
BASE_MODEL = "fastino/gliner2-base-v1"
# The packaged checkpoint lives next to bond_gliner2_pkg in this workspace.
# Keep the old test/models location as a fallback for downloaded snapshots.
PACKAGED_MODEL_DIR = WORKSPACE_ROOT / "bond_gliner2_model" / "models" / "gliner2-base-v1"
LOCAL_MODEL_DIR = TEST_ROOT / "models" / "gliner2-base-v1"


def resolve_base_model() -> str:
    """Prefer a complete local snapshot and avoid an unnecessary download."""
    for model_dir in (PACKAGED_MODEL_DIR, LOCAL_MODEL_DIR):
        if model_dir.is_dir() and (model_dir / "model.safetensors").exists():
            return str(model_dir)
    return BASE_MODEL

# Split ratios (by fingerprint connected components)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

# Fields with >5% presence in the sample (optional focused schema).
# Full 76-field schema is used by default for exact-match eval fairness.
CORE_FIELDS = [
    "volume",
    "bond_code",
    "buyer",
    "seller",
    "settlement_type",
    "yield",
    "settlement_date",
    "bond_name",
    "residual_maturity",
    "send_to",
    "serial_number",
    "send_type",
    "send_to_trader",
    "net_price",
    "bridge_institution",
    "send_from",
    "source",
    "rating",
    "buyer_send_type",
    "seller_send_type",
    "bridge_trader_name",
]

# Training defaults (override via CLI)
# Mac CPU: use smaller batch; GPU server can override via env.
NER_EPOCHS = 5
STRUCTURE_EPOCHS = 8
BATCH_SIZE = 2
EVAL_BATCH_SIZE = 4
ENCODER_LR = 1e-5
TASK_LR = 5e-4
GRAD_ACCUM = 8
MAX_LEN = 384
EVAL_STEPS = 200
LOGGING_STEPS = 20

# List-valued fields in this dataset
LIST_FIELDS = {
    "send_to",
    "send_from",
    "contact_info",
    "buyer_contact_info",
    "seller_contact_info",
}

# 训练档位：Mac / RTX 4060
PROFILES = {
    "mac": {
        "batch_size": 2,
        "grad_accum": 8,
        "eval_batch_size": 4,
        "fp16": False,
        "ner_epochs": 5,
        "structure_epochs": 8,
        "eval_steps": 200,
    },
    "gpu4060": {
        "batch_size": 4,
        "grad_accum": 4,
        "eval_batch_size": 8,
        "fp16": True,
        "ner_epochs": 5,
        "structure_epochs": 8,
        "eval_steps": 200,
    },
}


def resolve_data_dir(schema_mode: str = "full", data_variant: str | None = None) -> Path:
    """Return prepared data directory. variant can differ from schema (e.g. full_split)."""
    return DATA_DIR / (data_variant or schema_mode)
