#!/usr/bin/env python3
"""Stage-2: train GLiNER2 structure extraction, warm-started from NER checkpoint."""

from __future__ import annotations

import argparse
import json
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
    p = argparse.ArgumentParser(description="Train structure stage from NER")
    p.add_argument("--schema-mode", choices=["full", "core"], default="full")
    p.add_argument(
        "--data-variant",
        default=None,
        help="Prepared data subdir (default: same as schema-mode)",
    )
    p.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="NER checkpoint dir (default: outputs/ner/<schema>/best or latest)",
    )
    p.add_argument("--fallback-base", default=None)
    p.add_argument("--train-file", type=Path, default=None)
    p.add_argument("--eval-file", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=cfg.STRUCTURE_EPOCHS)
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


def resolve_checkpoint_dir(path: Path) -> Path:
    """Resolve a training output dir to an actual loadable checkpoint."""
    path = Path(path)
    if (path / "config.json").exists():
        return path
    for candidate in ("best", "checkpoint-best", "final"):
        p = path / candidate
        if (p / "config.json").exists():
            return p
    ckpts = [p for p in path.glob("checkpoint-*") if (p / "config.json").exists()]
    if ckpts:
        def step_key(p: Path) -> int:
            try:
                return int(p.name.split("-")[-1])
            except ValueError:
                return -1
        return sorted(ckpts, key=step_key)[-1]
    return path


def resolve_init(args: argparse.Namespace, data_variant: str) -> str:
    if args.init_from is not None:
        path = Path(args.init_from)
        if not path.exists():
            raise FileNotFoundError(path)
        return str(resolve_checkpoint_dir(path))

    ner_root = cfg.NER_DIR / data_variant
    resolved = resolve_checkpoint_dir(ner_root)
    if (resolved / "config.json").exists():
        return str(resolved)

    print(
        f"[warn] NER checkpoint not found under {ner_root}; "
        f"falling back to {args.fallback_base or cfg.resolve_base_model()}"
    )
    return args.fallback_base or cfg.resolve_base_model()


def main() -> None:
    args = parse_args()
    data_variant = args.data_variant or args.schema_mode
    data_dir = cfg.resolve_data_dir(args.schema_mode, data_variant)
    train_file = args.train_file or (data_dir / "structure_train_clean.jsonl")
    eval_file = args.eval_file or (data_dir / "structure_val_clean.jsonl")
    output_dir = args.output_dir or (cfg.STRUCTURE_DIR / data_variant)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_file.exists():
        raise FileNotFoundError(
            f"Missing {train_file}. Run prepare_data.py first."
        )

    field_contract = validate_training_fields(
        train_file,
        args.schema_mode,
        "structure",
        allow_missing_labels=args.allow_missing_field_labels,
    )

    init_path = resolve_init(args, data_variant)

    from gliner2 import GLiNER2
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
    from mac_utils import apply_mac_training_defaults

    print(f"Loading init model: {init_path}")
    model = GLiNER2.from_pretrained(init_path, map_location="cpu")

    train_cfg = TrainingConfig(
        output_dir=str(output_dir),
        experiment_name=f"bond_deal_structure_from_ner_{args.schema_mode}",
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
    )
    apply_mac_training_defaults(train_cfg)

    meta = {
        "stage": "structure_from_ner",
        "init_from": init_path,
        "train_file": str(train_file),
        "eval_file": str(eval_file),
        "schema_mode": args.schema_mode,
        "data_variant": data_variant,
        "field_contract": field_contract,
        "config": {k: getattr(train_cfg, k) for k in [
            "num_epochs", "max_steps", "batch_size", "gradient_accumulation_steps",
            "encoder_lr", "task_lr", "max_len", "seed", "fp16", "bf16",
            "gradient_checkpointing", "early_stopping", "early_stopping_patience",
        ]},
    }
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
    with open(output_dir / "training_result.json", "w", encoding="utf-8") as f:
        json.dump(result if isinstance(result, dict) else {"result": str(result)},
                  f, ensure_ascii=False, indent=2, default=str)
    print("Structure training done.")
    print(f"Best / latest checkpoint under: {output_dir}")


if __name__ == "__main__":
    main()
