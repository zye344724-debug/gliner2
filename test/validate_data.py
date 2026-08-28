#!/usr/bin/env python3
"""Validate prepared JSONL without loading any model."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i} invalid JSON: {e}") from e
    return rows


def check_ner_row(row: Dict[str, Any], path: Path, idx: int) -> List[str]:
    errs = []
    if "input" not in row or "output" not in row:
        errs.append(f"{path}:{idx} missing input/output")
        return errs
    out = row["output"]
    if "entities" not in out:
        errs.append(f"{path}:{idx} missing entities")
        return errs
    text = row["input"]
    for etype, mentions in out["entities"].items():
        if not isinstance(mentions, list):
            errs.append(f"{path}:{idx} entities[{etype}] not list")
            continue
        for m in mentions:
            if not isinstance(m, str):
                errs.append(f"{path}:{idx} entity mention not str: {etype}")
            elif m and m not in text:
                errs.append(f"{path}:{idx} mention not in text: {etype}={m!r}")
    return errs


def check_structure_row(row: Dict[str, Any], path: Path, idx: int) -> List[str]:
    errs = []
    if "input" not in row or "output" not in row:
        errs.append(f"{path}:{idx} missing input/output")
        return errs
    out = row["output"]
    if "json_structures" not in out:
        errs.append(f"{path}:{idx} missing json_structures")
        return errs
    text = row["input"]
    for si, st in enumerate(out["json_structures"]):
        deal = st.get("deal", st)
        if not isinstance(deal, dict):
            errs.append(f"{path}:{idx} structure[{si}] not dict")
            continue
        for fk, fv in deal.items():
            if isinstance(fv, dict) and "text" in fv:
                t, s, e = fv.get("text"), fv.get("start"), fv.get("end")
                if t and (s is None or e is None):
                    errs.append(f"{path}:{idx} span missing start/end: {fk}")
                elif t and not (0 <= s <= e <= len(text)):
                    errs.append(f"{path}:{idx} span OOB: {fk} [{s},{e}) len={len(text)}")
                elif t and text[s:e] != t:
                    errs.append(f"{path}:{idx} span mismatch: {fk} {text[s:e]!r} vs {t!r}")
            elif isinstance(fv, list):
                for item in fv:
                    if isinstance(item, dict) and "text" in item:
                        t, s, e = item["text"], item.get("start"), item.get("end")
                        if t and text[s:e] != t:
                            errs.append(f"{path}:{idx} list span mismatch: {fk}")
    return errs


def summarize_split(data_dir: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {"data_dir": str(data_dir), "splits": {}}
    for split in ("train", "val", "test"):
        ner_path = data_dir / f"ner_{split}_clean.jsonl"
        st_path = data_dir / f"structure_{split}_clean.jsonl"
        if not ner_path.exists() or not st_path.exists():
            report["splits"][split] = {"error": "missing files"}
            continue

        ner_rows = load_jsonl(ner_path)
        st_rows = load_jsonl(st_path)
        ner_errs: List[str] = []
        st_errs: List[str] = []
        for i, r in enumerate(ner_rows, 1):
            ner_errs.extend(check_ner_row(r, ner_path, i))
        for i, r in enumerate(st_rows, 1):
            st_errs.extend(check_structure_row(r, st_path, i))

        n_deals = sum(len(r["output"]["json_structures"]) for r in st_rows)
        multi = sum(1 for r in st_rows if len(r["output"]["json_structures"]) > 1)
        lens = [len(r["input"]) for r in st_rows]
        has_span = sum(
            1
            for r in st_rows
            for st in r["output"]["json_structures"]
            for v in st["deal"].values()
            if isinstance(v, dict) and "start" in v
        )

        report["splits"][split] = {
            "n_ner": len(ner_rows),
            "n_structure": len(st_rows),
            "n_deals": n_deals,
            "multi_deal_samples": multi,
            "input_len_avg": sum(lens) / len(lens) if lens else 0,
            "input_len_max": max(lens) if lens else 0,
            "structure_span_fields": has_span,
            "ner_errors": len(ner_errs),
            "structure_errors": len(st_errs),
            "sample_ner_errors": ner_errs[:5],
            "sample_structure_errors": st_errs[:5],
        }
    stats_path = data_dir / "split_stats.json"
    if stats_path.exists():
        report["split_stats"] = json.loads(stats_path.read_text(encoding="utf-8"))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Validate prepared datasets")
    p.add_argument(
        "--data-dirs",
        nargs="*",
        default=[
            str(cfg.DATA_DIR / "full"),
            str(cfg.DATA_DIR / "full_split"),
            str(cfg.DATA_DIR / "core"),
        ],
        help="Directories to validate",
    )
    p.add_argument("--out", type=Path, default=cfg.LOG_DIR / "validate_data.json")
    args = p.parse_args()

    all_reports = []
    total_errors = 0
    for d in args.data_dirs:
        path = Path(d)
        if not path.exists():
            all_reports.append({"data_dir": str(path), "skipped": "not found"})
            continue
        rep = summarize_split(path)
        for sp in rep.get("splits", {}).values():
            if isinstance(sp, dict):
                total_errors += sp.get("ner_errors", 0) + sp.get("structure_errors", 0)
        all_reports.append(rep)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"total_errors": total_errors, "reports": all_reports}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if total_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
