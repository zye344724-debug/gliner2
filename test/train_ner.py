#!/usr/bin/env python3
"""Stage-1: train GLiNER2 NER on bond-deal field mentions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
for path in (REPO_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import config as cfg
from field_contract import validate_training_fields


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NER stage")
    p.add_argument("--schema-mode", choices=["full", "core"], default="full")
    p.add_argument(
        "--data-variant",
        default=None,
        help="Prepared data subdir (default: same as schema-mode, e.g. full_split)",
    )
    p.add_argument("--base-model", default=None, help="HF repo or local dir; default auto")
    p.add_argument("--train-file", type=Path, default=None)
    p.add_argument("--eval-file", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=cfg.NER_EPOCHS)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--eval-batch-size", type=int, default=cfg.EVAL_BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=cfg.GRAD_ACCUM)
    p.add_argument("--encoder-lr", type=float, default=cfg.ENCODER_LR)
    p.add_argument("--task-lr", type=float, default=cfg.TASK_LR)
    p.add_argument("--max-len", type=int, default=cfg.MAX_LEN)
    p.add_argument("--eval-steps", type=int, default=cfg.EVAL_STEPS)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--bf16", action="store_true", default=False)
    p.add_argument("--max-train-samples", type=int, default=-1)
    p.add_argument("--max-eval-samples", type=int, default=-1)
    p.add_argument("--validate-data", action="store_true", default=False)
    p.add_argument("--gradient-checkpointing", action="store_true", default=False)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--allow-missing-field-labels",
        action="store_true",
        help="Diagnostic smoke runs only; formal training must cover every field",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    is_primary = int(os.environ.get("RANK", "0")) == 0
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    data_variant = args.data_variant or args.schema_mode
    data_dir = cfg.resolve_data_dir(args.schema_mode, data_variant)
    train_file = args.train_file or (data_dir / "ner_train_clean.jsonl")
    eval_file = args.eval_file or (data_dir / "ner_val_clean.jsonl")
    output_dir = args.output_dir or (cfg.NER_DIR / data_variant)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_file.exists():
        raise FileNotFoundError(
            f"Missing {train_file}. Run prepare_data.py first."
        )

    field_contract = validate_training_fields(
        train_file,
        args.schema_mode,
        "ner",
        allow_missing_labels=args.allow_missing_field_labels,
    )

    from gliner2 import GLiNER2
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
    from mac_utils import apply_mac_training_defaults

    base_model = args.base_model or cfg.resolve_base_model()
    print(f"Loading base model: {base_model}")
    model = GLiNER2.from_pretrained(base_model, map_location="cpu")

    # Entity-only NER rows never execute the classification or instance-count
    # predictors. Freeze those fixed-unused heads so DDP does not wait for
    # gradients that cannot exist in this stage. Their pretrained weights are
    # still saved and count_pred is trainable again when stage 2 reloads them.
    frozen_modules = ("classifier", "count_pred")
    for module_name in frozen_modules:
        module = getattr(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    if is_primary:
        print(f"Frozen unused NER heads: {', '.join(frozen_modules)}")

    train_cfg = TrainingConfig(
        output_dir=str(output_dir),
        experiment_name=f"bond_deal_ner_{args.schema_mode}",
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        encoder_lr=args.encoder_lr,
        task_lr=args.task_lr,
        max_len=args.max_len,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=cfg.LOGGING_STEPS,
        save_best=True,
        metric_for_best="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        validate_data=args.validate_data,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        num_workers=args.num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        early_stopping=args.early_stopping_patience > 0,
        early_stopping_patience=max(1, args.early_stopping_patience),
        save_total_limit=2,
        report_to_wandb=False,
        local_rank=local_rank,
        # PyTorch DDP static_graph is incompatible with a first backward under
        # no_sync(), which is how gradient accumulation starts in this trainer.
        ddp_static_graph=False if local_rank >= 0 else True,
    )
    apply_mac_training_defaults(train_cfg)

    meta = {
        "stage": "ner",
        "base_model": base_model,
        "train_file": str(train_file),
        "eval_file": str(eval_file),
        "schema_mode": args.schema_mode,
        "data_variant": data_variant,
        "field_contract": field_contract,
        "frozen_modules": list(frozen_modules),
        "config": {k: getattr(train_cfg, k) for k in [
            "num_epochs", "max_steps", "batch_size", "gradient_accumulation_steps",
            "encoder_lr", "task_lr", "max_len", "seed", "fp16", "bf16",
            "gradient_checkpointing", "early_stopping", "early_stopping_patience",
        ]},
    }
    if is_primary:
        with open(output_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    trainer = GLiNER2Trainer(model, train_cfg)
    print(f"Train: {train_file}")
    print(f"Eval : {eval_file}")
    print(f"Out  : {output_dir}")
    result = trainer.train(
        train_data=str(train_file),
        eval_data=str(eval_file) if eval_file.exists() else None,
    )
    if is_primary:
        with open(output_dir / "training_result.json", "w", encoding="utf-8") as f:
            json.dump(result if isinstance(result, dict) else {"result": str(result)},
                      f, ensure_ascii=False, indent=2, default=str)
    print("NER training done.")
    print(f"Best / latest checkpoint under: {output_dir}")


if __name__ == "__main__":
    main()
