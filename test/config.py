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

# Full-field auxiliary curricula.  The primary training rows always keep all
# 76 fields; these groups create additional, smaller-schema views for examples
# containing rare or semantically confusable labels.  Keeping confusable
# siblings together (buyer/seller, call/maturity yield, etc.) teaches the model
# the distinction instead of merely duplicating the same full-schema loss.
FIELD_FOCUS_GROUPS = {
    "bond_identity": {
        "bond_code", "bond_name", "bond_type", "residual_maturity", "rating",
    },
    "pricing": {
        "volume", "yield", "call_yield", "maturity_yield", "valuation",
        "net_price", "full_price", "fee", "buyer_fee", "seller_fee",
    },
    "settlement": {
        "settlement_type", "buyer_settlement_type", "seller_settlement_type",
        "settlement_date", "buyer_settlement_date", "seller_settlement_date",
        "date",
    },
    "parties": {
        "institution", "buyer", "seller", "bridge_institution",
        "contact_institution", "subject_name", "buyer_subject_name",
        "seller_subject_name", "subject_short_name", "buyer_subject_short_name",
        "seller_subject_short_name",
    },
    "people": {
        "trader_name", "buyer_trader_name", "seller_trader_name",
        "bridge_trader_name", "contact_person", "person_name", "sale_manager",
    },
    "delivery": {
        "send_type", "buyer_send_type", "seller_send_type", "send_to",
        "send_from", "send_to_trader", "send_from_trader", "contact_info",
        "buyer_contact_info", "seller_contact_info",
    },
    "accounts_and_codes": {
        "account", "buyer_account", "seller_account", "trader_code",
        "buyer_trader_code", "seller_trader_code", "trading_subject_code",
        "buyer_trading_subject_code", "seller_trading_subject_code",
        "trading_broker_code", "buyer_trading_broker_code",
        "seller_trading_broker_code", "trading_broker_name",
        "buyer_trading_broker_name", "seller_trading_broker_name",
        "seat_number", "buyer_seat_number", "seller_seat_number",
    },
    "workflow": {
        "serial_number", "order_number", "agreement_no", "sale_department",
        "source", "exchange", "trade_intent", "deal_update_action",
    },
}

# These fields are difficult because their label depends on direction, role, or
# an action cue rather than surface form alone.  They receive at least one focus
# view even when their raw frequency is above the automatic rare-field cutoff.
HARD_FIELDS = {
    "buyer", "seller", "institution", "bridge_institution",
    "buyer_settlement_type", "seller_settlement_type",
    "buyer_settlement_date", "seller_settlement_date",
    "send_type", "buyer_send_type", "seller_send_type",
    "send_to", "send_from", "send_to_trader", "send_from_trader",
    "buyer_account", "seller_account", "buyer_trader_name",
    "seller_trader_name", "bridge_trader_name", "call_yield",
    "maturity_yield", "buyer_fee", "seller_fee",
}

# Stable deal-identifying context included in every focused structure view.
FOCUS_ANCHOR_FIELDS = {
    "bond_code", "bond_name", "serial_number", "volume", "yield",
}

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
