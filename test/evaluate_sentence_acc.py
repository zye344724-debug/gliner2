#!/usr/bin/env python3
"""Evaluate sentence-level exact-match accuracy for structure extraction.

单句准确率定义:
  对测试集每一句，将预测与 gold 的 deal 列表规范化后做多重集合匹配；
  仅当所有 deal 的非空字段完全一致时记为 1，否则为 0。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
for path in (REPO_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import config as cfg
from bond_boundary_normalizer import normalize_field_boundary


def normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        # span form or choice form
        if "text" in value:
            return str(value["text"]).strip()
        if "value" in value:
            return str(value["value"]).strip()
        return {k: normalize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        items = [normalize_value(v) for v in value]
        items = [x for x in items if x not in (None, "", [])]
        # order-insensitive for list fields
        try:
            return tuple(sorted(items))
        except TypeError:
            return tuple(items)
    if isinstance(value, str):
        return value.strip()
    return value


def compact_deal(
    deal: Dict[str, Any], *, normalize_boundaries: bool = False
) -> Tuple[Tuple[str, Any], ...]:
    items = []
    for k, v in deal.items():
        nv = normalize_value(v)
        if normalize_boundaries:
            nv = normalize_field_boundary(k, nv)
        if nv in (None, "", (), []):
            continue
        items.append((k, nv))
    return tuple(sorted(items, key=lambda x: x[0]))


def deals_from_gold_output(output: Dict[str, Any]) -> List[Tuple[Tuple[str, Any], ...]]:
    deals = []
    for item in output.get("json_structures", []):
        deal = item.get("deal", item)
        if isinstance(deal, dict):
            compact = compact_deal(deal)
            if compact:
                deals.append(compact)
    return deals


def deals_from_prediction(
    pred: Dict[str, Any], *, normalize_boundaries: bool = True
) -> List[Tuple[Tuple[str, Any], ...]]:
    """Accept both official extract_json shapes and training shapes."""
    deals: List[Tuple[Tuple[str, Any], ...]] = []

    if not pred:
        return deals

    # Shape A: {"deal": [ {...}, {...} ]}
    if "deal" in pred and isinstance(pred["deal"], list):
        for deal in pred["deal"]:
            if isinstance(deal, dict):
                compact = compact_deal(
                    deal, normalize_boundaries=normalize_boundaries
                )
                if compact:
                    deals.append(compact)
        return deals

    # Shape B: {"json_structures": [{"deal": {...}}, ...]}
    if "json_structures" in pred:
        for item in pred.get("json_structures", []):
            deal = item.get("deal", item) if isinstance(item, dict) else None
            if isinstance(deal, dict):
                compact = compact_deal(
                    deal, normalize_boundaries=normalize_boundaries
                )
                if compact:
                    deals.append(compact)
        return deals

    # Shape C: nested under "structures" / raw list
    if isinstance(pred, list):
        for item in pred:
            if isinstance(item, dict) and "deal" in item:
                compact = compact_deal(
                    item["deal"], normalize_boundaries=normalize_boundaries
                )
            elif isinstance(item, dict):
                compact = compact_deal(
                    item, normalize_boundaries=normalize_boundaries
                )
            else:
                continue
            if compact:
                deals.append(compact)
        return deals

    return deals


def sentence_exact_match(
    gold_deals: Sequence[Tuple[Tuple[str, Any], ...]],
    pred_deals: Sequence[Tuple[Tuple[str, Any], ...]],
) -> bool:
    return Counter(gold_deals) == Counter(pred_deals)


def field_level_scores(
    gold_deals: Sequence[Tuple[Tuple[str, Any], ...]],
    pred_deals: Sequence[Tuple[Tuple[str, Any], ...]],
) -> Tuple[int, int, int]:
    """Greedy deal alignment then micro field TP/FP/FN (diagnostic only)."""
    remaining_pred = list(pred_deals)
    tp = fp = fn = 0
    used = [False] * len(remaining_pred)

    for g in gold_deals:
        g_map = dict(g)
        best_j = -1
        best_overlap = -1
        for j, p in enumerate(remaining_pred):
            if used[j]:
                continue
            p_map = dict(p)
            keys = set(g_map) | set(p_map)
            overlap = sum(1 for k in keys if g_map.get(k) == p_map.get(k))
            if overlap > best_overlap:
                best_overlap = overlap
                best_j = j
        if best_j < 0:
            fn += len(g_map)
            continue
        used[best_j] = True
        p_map = dict(remaining_pred[best_j])
        keys = set(g_map) | set(p_map)
        for k in keys:
            gv, pv = g_map.get(k, None), p_map.get(k, None)
            if gv is not None and pv is not None and gv == pv:
                tp += 1
            elif gv is not None and (pv is None or gv != pv):
                fn += 1
            elif gv is None and pv is not None:
                fp += 1
        # mismatched values already counted as fn; extra wrong values as fp
        for k in p_map:
            if k in g_map and g_map[k] != p_map[k]:
                fp += 1

    for j, p in enumerate(remaining_pred):
        if not used[j]:
            fp += len(dict(p))
    return tp, fp, fn


def per_field_scores(
    gold_deals: Sequence[Tuple[Tuple[str, Any], ...]],
    pred_deals: Sequence[Tuple[Tuple[str, Any], ...]],
) -> Dict[str, Counter]:
    """Greedy deal alignment with TP/FP/FN retained for every field."""
    scores: Dict[str, Counter] = defaultdict(Counter)
    used = [False] * len(pred_deals)
    for gold in gold_deals:
        gold_map = dict(gold)
        best_j = -1
        best_overlap = -1
        for j, pred in enumerate(pred_deals):
            if used[j]:
                continue
            pred_map = dict(pred)
            overlap = sum(
                1 for key in set(gold_map) | set(pred_map)
                if gold_map.get(key) == pred_map.get(key)
            )
            if overlap > best_overlap:
                best_overlap, best_j = overlap, j
        pred_map = dict(pred_deals[best_j]) if best_j >= 0 else {}
        if best_j >= 0:
            used[best_j] = True
        for key in set(gold_map) | set(pred_map):
            gold_value, pred_value = gold_map.get(key), pred_map.get(key)
            if gold_value is not None and gold_value == pred_value:
                scores[key]["tp"] += 1
            else:
                if gold_value is not None:
                    scores[key]["fn"] += 1
                if pred_value is not None:
                    scores[key]["fp"] += 1
    for j, pred in enumerate(pred_deals):
        if not used[j]:
            for key in dict(pred):
                scores[key]["fp"] += 1
    return scores


def validate_field_contract(
    rows: Sequence[Dict[str, Any]], descriptions: Dict[str, Dict[str, str]]
) -> Dict[str, Any]:
    """Ensure evaluation data exposes exactly the requested schema fields."""
    schema_fields = set(descriptions.get("deal", {}))
    declared_fields = set()
    gold_fields = set()
    observed_gold_fields = set()
    for row in rows:
        output = row.get("output", {})
        declared_fields.update(output.get("json_descriptions", {}).get("deal", {}))
        for item in output.get("json_structures", []):
            deal = item.get("deal", item)
            if isinstance(deal, dict):
                gold_fields.update(deal)
                observed_gold_fields.update(
                    key
                    for key, value in deal.items()
                    if normalize_value(value) not in (None, "", (), [])
                )
    missing_declared = sorted(schema_fields - declared_fields)
    unexpected_declared = sorted(declared_fields - schema_fields)
    unexpected_gold = sorted(gold_fields - schema_fields)
    if missing_declared or unexpected_declared or unexpected_gold:
        raise ValueError(
            "Evaluation field contract failed: "
            f"missing_declared={missing_declared}, "
            f"unexpected_declared={unexpected_declared}, "
            f"unexpected_gold={unexpected_gold}"
        )
    return {
        "status": "passed",
        "schema_field_count": len(schema_fields),
        "declared_field_count": len(declared_fields),
        "gold_field_count": len(gold_fields),
        "observed_non_null_gold_field_count": len(observed_gold_fields),
        "fields_without_non_null_gold_support": sorted(
            schema_fields - observed_gold_fields
        ),
        "all_fields_have_non_null_gold_support": (
            observed_gold_fields == schema_fields
        ),
        "fields": sorted(schema_fields),
    }


def build_structure_schema(descriptions: Dict[str, Dict[str, str]], list_fields: set) -> Dict[str, List[str]]:
    fields = []
    for name, desc in descriptions.get("deal", {}).items():
        dtype = "list" if name in list_fields else "str"
        fields.append(f"{name}::{dtype}::{desc}")
    return {"deal": fields}


def _filter_confident_value(value: Any, threshold: float) -> Any:
    """Drop confidence-bearing values below a field-specific threshold."""
    if isinstance(value, list):
        kept = [_filter_confident_value(item, threshold) for item in value]
        return [item for item in kept if item not in (None, "", [], ())]
    if isinstance(value, dict):
        confidence = value.get("confidence")
        if confidence is not None and float(confidence) < threshold:
            return None
        return dict(value)
    return value


def filter_prediction_by_field_thresholds(
    prediction: Any,
    thresholds: Dict[str, float],
    default_threshold: float,
) -> Any:
    """Apply per-field thresholds to formatted confidence predictions."""
    if isinstance(prediction, list):
        return [
            filter_prediction_by_field_thresholds(item, thresholds, default_threshold)
            for item in prediction
        ]
    if not isinstance(prediction, dict):
        return prediction

    def filter_deal(deal: Any) -> Any:
        if not isinstance(deal, dict):
            return deal
        return {
            field: _filter_confident_value(
                value, thresholds.get(field, default_threshold)
            )
            for field, value in deal.items()
        }

    result = dict(prediction)
    if isinstance(result.get("deal"), list):
        result["deal"] = [filter_deal(deal) for deal in result["deal"]]
    if isinstance(result.get("json_structures"), list):
        structures = []
        for item in result["json_structures"]:
            if isinstance(item, dict) and "deal" in item:
                copied = dict(item)
                copied["deal"] = filter_deal(item["deal"])
                structures.append(copied)
            else:
                structures.append(item)
        result["json_structures"] = structures
    return result


def _field_metric(
    rows, predictions, field: str, normalize_boundaries: bool = True
) -> Dict[str, float]:
    counts: Counter = Counter()
    for row, prediction in zip(rows, predictions):
        gold = deals_from_gold_output(row["output"])
        pred = deals_from_prediction(
            prediction, normalize_boundaries=normalize_boundaries
        )
        counts.update(per_field_scores(gold, pred).get(field, {}))
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "support": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def tune_field_thresholds(
    rows,
    confidence_predictions,
    fields,
    candidates,
    default_threshold,
    normalize_boundaries=True,
):
    """Tune thresholds on validation without repeating model inference.

    First maximize each field's validation F1, then keep only coordinate changes
    that improve strict sentence exact match. All operations use cached
    confidence-bearing predictions produced at the lowest candidate threshold.
    """
    thresholds = {field: default_threshold for field in fields}
    diagnostics: Dict[str, Any] = {"field_scores": {}}
    for field in tqdm(
        fields,
        desc="Field F1 tuning",
        unit="field",
        dynamic_ncols=True,
        mininterval=1.0,
    ):
        scores = {}
        for candidate in candidates:
            trial = dict(thresholds)
            trial[field] = candidate
            filtered = [
                filter_prediction_by_field_thresholds(
                    prediction, trial, default_threshold
                )
                for prediction in confidence_predictions
            ]
            scores[str(candidate)] = _field_metric(
                rows, filtered, field, normalize_boundaries
            )
        support = max((metric["support"] for metric in scores.values()), default=0)
        if support:
            chosen = max(
                candidates,
                key=lambda candidate: (
                    scores[str(candidate)]["f1"],
                    scores[str(candidate)]["precision"],
                    -abs(candidate - default_threshold),
                ),
            )
        else:
            # With no validation positives, suppress false positives.
            chosen = max(
                candidates,
                key=lambda candidate: (-scores[str(candidate)]["fp"], candidate),
            )
        thresholds[field] = chosen
        diagnostics["field_scores"][field] = {
            "selected": chosen,
            "candidates": scores,
        }

    filtered = [
        filter_prediction_by_field_thresholds(pred, thresholds, default_threshold)
        for pred in confidence_predictions
    ]
    current_exact = exact_accuracy(rows, filtered, normalize_boundaries)
    diagnostics["exact_after_field_f1_tuning"] = current_exact
    # One cheap coordinate pass aligns field-F1 choices with the business metric.
    for field in tqdm(
        fields,
        desc="Exact-match tuning",
        unit="field",
        dynamic_ncols=True,
        mininterval=1.0,
    ):
        current = thresholds[field]
        best, best_exact = current, current_exact
        for candidate in candidates:
            if candidate == current:
                continue
            trial = dict(thresholds)
            trial[field] = candidate
            trial_predictions = [
                filter_prediction_by_field_thresholds(
                    prediction, trial, default_threshold
                )
                for prediction in confidence_predictions
            ]
            score = exact_accuracy(
                rows, trial_predictions, normalize_boundaries
            )
            if score > best_exact:
                best, best_exact = candidate, score
        thresholds[field] = best
        current_exact = best_exact
    diagnostics["exact_after_coordinate_tuning"] = current_exact
    return thresholds, diagnostics


def resolve_model_dir(path: Optional[Path], data_variant: str) -> Path:
    if path is not None:
        root = Path(path)
    else:
        root = cfg.STRUCTURE_DIR / data_variant

    if (root / "config.json").exists():
        return root
    for candidate in ("best", "checkpoint-best", "final"):
        p = root / candidate
        if (p / "config.json").exists():
            return p
    ckpts = [p for p in root.glob("checkpoint-*") if (p / "config.json").exists()]
    if ckpts:
        def step_key(p: Path) -> int:
            try:
                return int(p.name.split("-")[-1])
            except ValueError:
                return -1
        return sorted(ckpts, key=step_key)[-1]
    raise FileNotFoundError(f"No structure checkpoint under {root}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sentence exact-match evaluation")
    p.add_argument("--schema-mode", choices=["full", "core"], default="full")
    p.add_argument(
        "--allow-partial-schema",
        action="store_true",
        help="Permit diagnostic core-schema evaluation; never business-compliant",
    )
    p.add_argument("--data-variant", default=None)
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--test-file", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tune-thresholds", default="")
    p.add_argument(
        "--tune-field-thresholds",
        default="",
        help=(
            "Comma-separated per-field thresholds. Runs inference once at the "
            "lowest candidate and tunes on validation only"
        ),
    )
    p.add_argument("--tune-file", type=Path, default=None)
    p.add_argument("--tune-limit", type=int, default=300)
    p.add_argument("--max-len", type=int, default=cfg.MAX_LEN)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, cuda:N, or mps (default: auto)",
    )
    p.add_argument(
        "--no-boundary-normalization",
        action="store_true",
        help="Disable bond-specific cue-word boundary cleanup",
    )
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def load_rows(path: Path, limit: int = -1) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def run_inference(
    model,
    rows,
    schema,
    threshold,
    max_len,
    batch_size,
    include_confidence=False,
    desc="Inference",
):
    texts = [r["input"] for r in rows]
    preds = []
    bs = max(1, batch_size)
    starts = range(0, len(texts), bs)
    for i in tqdm(
        starts,
        total=(len(texts) + bs - 1) // bs,
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ):
        chunk = texts[i : i + bs]
        if hasattr(model, "batch_extract_json"):
            chunk_preds = model.batch_extract_json(
                chunk,
                schema,
                batch_size=bs,
                threshold=threshold,
                max_len=max_len,
                include_confidence=include_confidence,
            )
        else:
            chunk_preds = [
                model.extract_json(
                    t,
                    schema,
                    threshold=threshold,
                    max_len=max_len,
                    include_confidence=include_confidence,
                )
                for t in chunk
            ]
        preds.extend(chunk_preds)
    return preds


def exact_accuracy(rows, preds, normalize_boundaries: bool = True) -> float:
    if not rows:
        return 0.0
    return sum(
        sentence_exact_match(
            deals_from_gold_output(row["output"]),
            deals_from_prediction(
                pred, normalize_boundaries=normalize_boundaries
            ),
        )
        for row, pred in zip(rows, preds)
    ) / len(rows)


def main() -> None:
    args = parse_args()
    normalize_boundaries = not args.no_boundary_normalization
    if args.schema_mode != "full" and not args.allow_partial_schema:
        raise ValueError(
            "Partial-schema evaluation is diagnostic only. Formal business "
            "evaluation must use --schema-mode full. Pass --allow-partial-schema "
            "only when a deliberately non-business core metric is required."
        )
    data_variant = args.data_variant or args.schema_mode
    data_dir = cfg.resolve_data_dir(args.schema_mode, data_variant)
    test_file = args.test_file or (data_dir / "structure_test_clean.jsonl")
    model_dir = resolve_model_dir(args.model_dir, data_variant)
    out_path = args.out
    if out_path is None:
        model_root = Path(model_dir)
        if "smoke" in str(model_root):
            out_path = model_root / "eval_sentence_acc.json"
        else:
            out_path = cfg.STRUCTURE_DIR / data_variant / "eval_sentence_acc.json"

    with open(cfg.SCHEMA_DESC, encoding="utf-8") as f:
        descriptions = json.load(f)
    if args.schema_mode == "core":
        descriptions = {
            "deal": {
                k: v
                for k, v in descriptions["deal"].items()
                if k in cfg.CORE_FIELDS
            }
        }

    rows = load_rows(test_file, args.limit)
    field_contract = validate_field_contract(rows, descriptions)

    from gliner2 import GLiNER2
    import torch

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA evaluation requested but CUDA is unavailable: {device}")

    print(f"Loading model: {model_dir}")
    print(f"Evaluation device: {device}")
    model = GLiNER2.from_pretrained(str(model_dir), map_location=device)
    model.eval()
    schema = build_structure_schema(descriptions, cfg.LIST_FIELDS)

    threshold = args.threshold
    tuning_scores = {}
    tune_rows = None
    if args.tune_thresholds.strip():
        tune_file = args.tune_file or (data_dir / "structure_val_clean.jsonl")
        tune_rows = load_rows(tune_file, args.tune_limit)
        candidates = [float(x.strip()) for x in args.tune_thresholds.split(",") if x.strip()]
        if not candidates:
            raise ValueError("--tune-thresholds did not contain a number")
        for candidate in candidates:
            tune_preds = run_inference(
                model,
                tune_rows,
                schema,
                candidate,
                args.max_len,
                args.batch_size,
                desc=f"Validation threshold {candidate:.2f}",
            )
            tuning_scores[str(candidate)] = exact_accuracy(
                tune_rows, tune_preds, normalize_boundaries
            )
            print(f"validation threshold={candidate:.3f} exact={tuning_scores[str(candidate)]:.4f}")
        threshold = max(
            candidates,
            key=lambda x: (tuning_scores[str(x)], -abs(x - args.threshold)),
        )
        print(f"selected threshold={threshold:.3f}")

    field_thresholds: Dict[str, float] = {}
    field_tuning_diagnostics: Dict[str, Any] = {}
    field_candidates = [
        float(value.strip())
        for value in args.tune_field_thresholds.split(",")
        if value.strip()
    ]
    if field_candidates:
        if any(not 0.0 < candidate < 1.0 for candidate in field_candidates):
            raise ValueError("--tune-field-thresholds values must be in (0, 1)")
        field_candidates = sorted(set(field_candidates))
        if tune_rows is None:
            tune_file = args.tune_file or (data_dir / "structure_val_clean.jsonl")
            tune_rows = load_rows(tune_file, args.tune_limit)
        confidence_predictions = run_inference(
            model,
            tune_rows,
            schema,
            field_candidates[0],
            args.max_len,
            args.batch_size,
            include_confidence=True,
            desc="Validation confidence inference",
        )
        baseline_predictions = [
            filter_prediction_by_field_thresholds(pred, {}, threshold)
            for pred in confidence_predictions
        ]
        field_tuning_diagnostics["candidate_floor"] = field_candidates[0]
        field_tuning_diagnostics["exact_before_field_tuning"] = exact_accuracy(
            tune_rows, baseline_predictions, normalize_boundaries
        )
        field_thresholds, tuned = tune_field_thresholds(
            tune_rows,
            confidence_predictions,
            sorted(descriptions.get("deal", {})),
            field_candidates,
            threshold,
            normalize_boundaries,
        )
        field_tuning_diagnostics.update(tuned)
        print(
            "field threshold tuning exact: "
            f"{field_tuning_diagnostics['exact_before_field_tuning']:.4f} -> "
            f"{field_tuning_diagnostics['exact_after_coordinate_tuning']:.4f}"
        )

    n_correct = 0
    field_tp = field_fp = field_fn = 0
    details = []
    field_counts: Dict[str, Counter] = defaultdict(Counter)

    if field_thresholds:
        raw_preds = run_inference(
            model,
            rows,
            schema,
            min(field_candidates),
            args.max_len,
            args.batch_size,
            include_confidence=True,
            desc="Test confidence inference",
        )
        preds = [
            filter_prediction_by_field_thresholds(
                prediction, field_thresholds, threshold
            )
            for prediction in raw_preds
        ]
    else:
        preds = run_inference(
            model,
            rows,
            schema,
            threshold,
            args.max_len,
            args.batch_size,
            desc="Test inference",
        )

    for row, pred in zip(rows, preds):
        gold_deals = deals_from_gold_output(row["output"])
        pred_deals = deals_from_prediction(
            pred, normalize_boundaries=normalize_boundaries
        )
        ok = sentence_exact_match(gold_deals, pred_deals)
        n_correct += int(ok)
        tp, fp, fn = field_level_scores(gold_deals, pred_deals)
        field_tp += tp
        field_fp += fp
        field_fn += fn
        row_field_counts = per_field_scores(gold_deals, pred_deals)
        for field, counts in row_field_counts.items():
            field_counts[field].update(counts)
        error_fields = {
            field: dict(counts)
            for field, counts in sorted(row_field_counts.items())
            if counts["fp"] or counts["fn"]
        }
        details.append(
            {
                "input": row["input"],
                "exact": ok,
                "n_gold_deals": len(gold_deals),
                "n_pred_deals": len(pred_deals),
                "error_fields": error_fields,
                "gold_deals": [dict(deal) for deal in gold_deals],
                "pred_deals": [dict(deal) for deal in pred_deals],
            }
        )

    n = len(rows)
    precision = field_tp / (field_tp + field_fp) if (field_tp + field_fp) else 0.0
    recall = field_tp / (field_tp + field_fn) if (field_tp + field_fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    summary = {
        "model_dir": str(model_dir),
        "test_file": str(test_file),
        "n_samples": n,
        "sentence_exact_match": n_correct / n if n else 0.0,
        "sentence_correct": n_correct,
        "field_micro_precision": precision,
        "field_micro_recall": recall,
        "field_micro_f1": f1,
        "threshold": threshold,
        "boundary_normalization": normalize_boundaries,
        "validation_threshold_scores": tuning_scores,
        "field_thresholds": field_thresholds,
        "field_threshold_tuning": field_tuning_diagnostics,
        "schema_mode": args.schema_mode,
        "data_variant": data_variant,
        "business_full_field_evaluation": args.schema_mode == "full",
        "business_field_coverage_ready": (
            args.schema_mode == "full"
            and field_contract["all_fields_have_non_null_gold_support"]
        ),
        "field_contract": field_contract,
    }

    field_metrics = {}
    for field in sorted(descriptions.get("deal", {})):
        counts = field_counts[field]
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        field_metrics[field] = {
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "field_metrics": field_metrics, "details": details},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
