#!/usr/bin/env python3
"""Convert bond-deal JSONL into GLiNER2 NER / structure splits.

Steps:
1. Strip null fields; convert {text,start,end} -> text (or list[str]).
2. Inject unified json_descriptions for every structure sample.
3. Flatten structure spans into NER entities (field name -> mentions).
4. Split by fingerprint connected components to avoid augmentation leakage.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg


def load_descriptions() -> Dict[str, Dict[str, str]]:
    with open(cfg.SCHEMA_DESC, encoding="utf-8") as f:
        return json.load(f)


def collect_raw_field_counts(samples: Iterable[Dict[str, Any]]) -> Tuple[Counter, Counter]:
    """Return deal-field occurrence and non-null occurrence counts."""
    present: Counter = Counter()
    non_null: Counter = Counter()
    for sample in samples:
        for item in sample.get("output", {}).get("json_structures", []):
            deal = item.get("deal", item)
            if not isinstance(deal, dict):
                continue
            for key, value in deal.items():
                present[key] += 1
                if value not in (None, "", []):
                    non_null[key] += 1
    return present, non_null


def validate_full_schema(
    samples: Iterable[Dict[str, Any]], descriptions: Dict[str, Dict[str, str]]
) -> Dict[str, Any]:
    """Fail closed when the business schema and raw labels diverge."""
    present, non_null = collect_raw_field_counts(samples)
    raw_fields = set(present)
    schema_fields = set(descriptions.get("deal", {}))
    raw_not_in_schema = sorted(raw_fields - schema_fields)
    schema_not_in_raw = sorted(schema_fields - raw_fields)
    schema_without_labels = sorted(k for k in schema_fields if non_null[k] == 0)
    if raw_not_in_schema or schema_not_in_raw:
        raise ValueError(
            "Full-field contract failed: "
            f"raw_not_in_schema={raw_not_in_schema}, "
            f"schema_not_in_raw={schema_not_in_raw}"
        )
    return {
        "status": "passed",
        "schema_field_count": len(schema_fields),
        "raw_field_count": len(raw_fields),
        "raw_not_in_schema": raw_not_in_schema,
        "schema_not_in_raw": schema_not_in_raw,
        "schema_without_non_null_labels": schema_without_labels,
        "all_fields_have_non_null_labels": not schema_without_labels,
        "non_null_labels_by_field": {
            key: non_null[key] for key in sorted(schema_fields)
        },
    }


def span_to_text(value: Any) -> Any:
    """Convert span dict / list-of-span to plain text values."""
    if value is None:
        return None
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict) and "text" in item:
                out.append(item["text"])
            elif isinstance(item, str):
                out.append(item)
            else:
                out.append(item)
        return out
    if isinstance(value, dict) and "text" in value:
        return value["text"]
    return value


def span_to_spanvalue(value: Any) -> Any:
    """Keep span dicts (text/start/end) intact; drop empty-text spans.

    Used by the structure route so the GLiNER2 processor can map char offsets
    to word spans precisely instead of surface string matching.
    """
    if value is None:
        return None
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict) and "text" in item:
                if item.get("text"):
                    out.append(item)
            elif isinstance(item, str):
                out.append(item)
            else:
                out.append(item)
        return out
    if isinstance(value, dict) and "text" in value:
        if value.get("text"):
            return value
        return None
    return value


def compact_deal(
    deal: Dict[str, Any],
    keep_fields: Optional[Set[str]] = None,
    keep_offsets: bool = False,
    keep_null: bool = False,
) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key, value in deal.items():
        if keep_fields is not None and key not in keep_fields:
            continue
        if value is None:
            if keep_null:
                compact[key] = None
            continue
        text_value = span_to_spanvalue(value) if keep_offsets else span_to_text(value)
        if text_value is None:
            if keep_null:
                compact[key] = None
            continue
        if text_value == "" or text_value == []:
            if keep_null:
                compact[key] = None
            continue
        compact[key] = text_value
    return compact


def structure_record(
    sample: Dict[str, Any],
    descriptions: Dict[str, Dict[str, str]],
    keep_fields: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    structures = []
    for item in sample["output"].get("json_structures", []):
        deal = item.get("deal", item)
        compact = compact_deal(deal, keep_fields=keep_fields, keep_offsets=True, keep_null=True)
        if compact and any(v is not None for v in compact.values()):
            structures.append({"deal": compact})
    return {
        "input": sample["input"],
        "output": {
            "json_structures": structures,
            "json_descriptions": descriptions,
        },
        "fingerprint": sample.get("fingerprint"),
        "meta": {
            "augmentation": sample.get("augmentation"),
            "n_deals": len(structures),
        },
    }


def ner_record(
    sample: Dict[str, Any],
    descriptions: Dict[str, Dict[str, str]],
    keep_fields: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    entities: Dict[str, List[str]] = defaultdict(list)
    seen: Dict[str, Set[str]] = defaultdict(set)
    for item in sample["output"].get("json_structures", []):
        deal = item.get("deal", item)
        compact = compact_deal(deal, keep_fields=keep_fields)
        for key, value in compact.items():
            values = value if isinstance(value, list) else [value]
            for mention in values:
                if not isinstance(mention, str):
                    mention = str(mention)
                if mention in seen[key]:
                    continue
                seen[key].add(mention)
                entities[key].append(mention)

    entity_descriptions = {
        k: descriptions.get("deal", {}).get(k, k) for k in entities.keys()
    }
    return {
        "input": sample["input"],
        "output": {
            "entities": dict(entities),
            "entity_descriptions": entity_descriptions,
        },
        "fingerprint": sample.get("fingerprint"),
        "meta": {
            "augmentation": sample.get("augmentation"),
            "n_entity_types": len(entities),
            "n_mentions": sum(len(v) for v in entities.values()),
        },
    }


def _iter_field_spans(value: Any):
    if isinstance(value, dict):
        if "start" in value and "end" in value:
            yield value["start"], value["end"]
        else:
            for v in value.values():
                yield from _iter_field_spans(v)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_field_spans(item)


def _remap_field(value: Any, offset: int, parent_end: int) -> Any:
    """Shift span offsets into sub-text; drop spans outside the parent interval."""
    if isinstance(value, dict) and "start" in value and "end" in value:
        s, e = value["start"], value["end"]
        if e <= offset or s >= parent_end:
            return None
        return {
            "text": value["text"],
            "start": s - offset,
            "end": e - offset,
        }
    if isinstance(value, list):
        items = []
        for item in value:
            mapped = _remap_field(item, offset, parent_end)
            if mapped is not None:
                items.append(mapped)
        return items if items else None
    return value


def split_sentence(text: str, deals: List[Dict[str, Any]]) -> Optional[List[Tuple[str, Dict[str, Any]]]]:
    """Split a multi-deal sentence into single-deal sub-sentences.

    Each deal's field spans define a contiguous char interval; we cut the text on
    those intervals (recomputing char offsets) and return one
    ``(sub_text, {"deal": {...}})`` per deal. Returns ``None`` when intervals
    overlap (shared trailing fields across deals) — those stay unsplit.
    """
    intervals: List[Tuple[int, int]] = []
    for it in deals:
        deal = it.get("deal", it)
        spans = list(_iter_field_spans(deal))
        if not spans:
            return None
        intervals.append((min(s for s, _ in spans), max(e for _, e in spans)))

    order = sorted(range(len(intervals)), key=lambda i: intervals[i][0])
    for a, b in zip(order, order[1:]):
        if intervals[b][0] < intervals[a][1]:
            return None

    results = []
    for i in order:
        s, e = intervals[i]
        sub_text = text[s:e]
        deal = deals[i].get("deal", deals[i])
        sub_deal: Dict[str, Any] = {}
        for k, v in deal.items():
            if v is None:
                sub_deal[k] = None
                continue
            mapped = _remap_field(v, s, e)
            sub_deal[k] = mapped if mapped not in (None, []) else None
        if any(v is not None for v in sub_deal.values()):
            results.append((sub_text, {"deal": sub_deal}))
    return results


def split_multi_deal(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a (possibly multi-deal) sample into single-deal samples.

    Single-deal and unsplittable samples are returned unchanged. Split sub-samples
    inherit the parent fingerprint with a ``#splitN`` suffix so the leakage-aware
    ``group_key`` (which strips after ``#``) keeps them grouped with their parent.
    """
    deals = sample["output"].get("json_structures", [])
    if len(deals) <= 1:
        return [sample]

    sub = split_sentence(sample["input"], deals)
    if sub is None:
        return [sample]

    fp = sample.get("fingerprint") or ""
    out = []
    for j, (st, sd) in enumerate(sub):
        out.append({
            "input": st,
            "output": {
                "json_structures": [sd],
                "json_descriptions": sample["output"].get("json_descriptions", {}),
            },
            "fingerprint": f"{fp}#split{j}" if fp else None,
            "augmentation": sample.get("augmentation"),
        })
    return out


def group_key(sample: Dict[str, Any], idx: int) -> str:
    """Leakage-aware group id without chaining multi_merge overlaps.

    - Prefer frozenset(source_fingerprints) so a multi_merge combo stays together
      but does not union with other combos that share one source.
    - Else use fingerprint stem (before '#').
    """
    aug = sample.get("augmentation") or {}
    sources = [str(s) for s in (aug.get("source_fingerprints") or []) if s]
    if sources:
        return "src::" + "||".join(sorted(set(sources)))
    fp = sample.get("fingerprint") or ""
    if fp:
        return "fp::" + fp.split("#")[0]
    return f"orphan::{idx}"


def build_groups(samples: List[Dict[str, Any]]) -> List[List[int]]:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, sample in enumerate(samples):
        buckets[group_key(sample, i)].append(i)
    return list(buckets.values())


def split_groups(
    groups: List[List[int]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[int], List[int], List[int]]:
    """Assign each connected component to the split furthest below target.

    Large multi_merge components must stay intact; greedy fill-train-first
    would starve val/test, so we balance remaining capacity instead.
    """
    rng = random.Random(seed)
    # Shuffle then stable-sort by size descending for better packing.
    indexed = list(enumerate(groups))
    rng.shuffle(indexed)
    indexed.sort(key=lambda x: len(x[1]), reverse=True)

    total = sum(len(g) for g in groups)
    targets = {
        "train": total * train_ratio,
        "val": total * val_ratio,
        "test": total * (1.0 - train_ratio - val_ratio),
    }
    buckets: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for _, idxs in indexed:
        # Prefer the split with the largest remaining fractional deficit.
        def deficit(name: str) -> float:
            if targets[name] <= 0:
                return -1e9
            return (targets[name] - counts[name]) / targets[name]

        choice = max(("train", "val", "test"), key=deficit)
        buckets[choice].extend(idxs)
        counts[choice] += len(idxs)

    return buckets["train"], buckets["val"], buckets["test"]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def summarize(
    name: str,
    indices: List[int],
    samples: List[Dict[str, Any]],
    schema_fields: Set[str],
) -> Dict[str, Any]:
    n_deals = 0
    aug = Counter()
    field_labels: Counter = Counter()
    for i in indices:
        n_deals += len(samples[i]["output"].get("json_structures", []))
        for item in samples[i]["output"].get("json_structures", []):
            deal = item.get("deal", item)
            if isinstance(deal, dict):
                for key, value in deal.items():
                    if value not in (None, "", []):
                        field_labels[key] += 1
        actions = (samples[i].get("augmentation") or {}).get("actions") or []
        if actions:
            aug[actions[0].get("type", "unknown")] += 1
        else:
            aug["(none)"] += 1
    return {
        "split": name,
        "n_samples": len(indices),
        "n_deals": n_deals,
        "aug_first_action": dict(aug.most_common()),
        "fields_with_non_null_labels": len(field_labels),
        "fields_without_non_null_labels": sorted(schema_fields - set(field_labels)),
        "non_null_labels_by_field": {
            key: field_labels[key] for key in sorted(schema_fields)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NER/structure datasets")
    parser.add_argument("--raw", type=Path, default=cfg.RAW_DATA)
    parser.add_argument("--out-dir", type=Path, default=cfg.DATA_DIR)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--train-ratio", type=float, default=cfg.TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=cfg.VAL_RATIO)
    parser.add_argument(
        "--schema-mode",
        choices=["full", "core"],
        default="full",
        help="full=all non-null fields; core=high-frequency fields only",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Optional cap for smoke tests",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Output subdir name under --out-dir (default: schema-mode or full_split)",
    )
    parser.add_argument(
        "--split-multi",
        action="store_true",
        default=False,
        help="Split multi-deal sentences into single-deal sub-samples",
    )
    args = parser.parse_args()

    variant = args.variant
    if variant is None and args.split_multi and args.schema_mode == "full":
        variant = "full_split"
    if variant is None:
        variant = args.schema_mode

    keep_fields: Optional[Set[str]] = None
    if args.schema_mode == "core":
        keep_fields = set(cfg.CORE_FIELDS)

    descriptions = load_descriptions()
    if keep_fields is not None:
        descriptions = {
            "deal": {
                k: v
                for k, v in descriptions.get("deal", {}).items()
                if k in keep_fields
            }
        }

    samples: List[Dict[str, Any]] = []
    with open(args.raw, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            if args.max_samples > 0 and len(samples) >= args.max_samples:
                break

    # The formal business route is deliberately fail-closed: adding a raw field
    # without adding its description (or silently losing one) must stop the run.
    field_contract = None
    if args.schema_mode == "full":
        field_contract = validate_full_schema(samples, descriptions)

    if args.split_multi:
        expanded: List[Dict[str, Any]] = []
        for s in samples:
            expanded.extend(split_multi_deal(s))
        samples = expanded

    groups = build_groups(samples)
    train_idx, val_idx, test_idx = split_groups(
        groups, args.seed, args.train_ratio, args.val_ratio
    )

    out_dir = args.out_dir / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    stats = {
        "raw": str(args.raw),
        "schema_mode": args.schema_mode,
        "variant": variant,
        "seed": args.seed,
        "n_groups": len(groups),
        "n_samples": len(samples),
        "business_full_field_evaluation": args.schema_mode == "full",
        "field_contract": field_contract,
        "splits": {},
    }

    for split_name, indices in split_map.items():
        ner_rows = [
            ner_record(samples[i], descriptions, keep_fields=keep_fields)
            for i in indices
        ]
        struct_rows = [
            structure_record(samples[i], descriptions, keep_fields=keep_fields)
            for i in indices
        ]
        # Drop empty structures (should be rare)
        struct_rows = [r for r in struct_rows if r["output"]["json_structures"]]
        ner_rows = [r for r in ner_rows if r["output"]["entities"]]

        n_ner = write_jsonl(out_dir / f"ner_{split_name}.jsonl", ner_rows)
        n_st = write_jsonl(out_dir / f"structure_{split_name}.jsonl", struct_rows)
        # Keep fingerprints for eval alignment
        write_jsonl(
            out_dir / f"meta_{split_name}.jsonl",
            [
                {
                    "fingerprint": samples[i].get("fingerprint"),
                    "input": samples[i]["input"],
                    "n_deals": len(samples[i]["output"].get("json_structures", [])),
                }
                for i in indices
            ],
        )
        stats["splits"][split_name] = {
            **summarize(
                split_name,
                indices,
                samples,
                set(descriptions.get("deal", {})),
            ),
            "n_ner_rows": n_ner,
            "n_structure_rows": n_st,
        }

    # Official-format training files without meta (cleaner for trainer)
    for split_name in split_map:
        for task in ("ner", "structure"):
            src = out_dir / f"{task}_{split_name}.jsonl"
            dst = out_dir / f"{task}_{split_name}_clean.jsonl"
            with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
                for line in fin:
                    row = json.loads(line)
                    fout.write(
                        json.dumps(
                            {"input": row["input"], "output": row["output"]},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    stats_path = out_dir / "split_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nWrote datasets under: {out_dir}")


if __name__ == "__main__":
    main()
