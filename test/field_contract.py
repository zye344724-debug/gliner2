"""Fail-closed field coverage checks shared by the bond training scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Set

import config as cfg


def expected_fields(schema_mode: str) -> Set[str]:
    descriptions = json.loads(cfg.SCHEMA_DESC.read_text(encoding="utf-8"))
    fields = set(descriptions.get("deal", {}))
    return fields if schema_mode == "full" else fields & set(cfg.CORE_FIELDS)


def validate_training_fields(
    path: Path,
    schema_mode: str,
    task: str,
    allow_missing_labels: bool = False,
) -> Dict[str, Any]:
    """Validate declared fields and positive-label coverage before model loading."""
    expected = expected_fields(schema_mode)
    declared: Set[str] = set()
    positive: Set[str] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            output = json.loads(line).get("output", {})
            if task == "ner":
                declared.update(output.get("entity_descriptions", {}))
                positive.update(
                    key for key, value in output.get("entities", {}).items() if value
                )
            elif task == "structure":
                declared.update(output.get("json_descriptions", {}).get("deal", {}))
                for item in output.get("json_structures", []):
                    deal = item.get("deal", item)
                    if isinstance(deal, dict):
                        positive.update(
                            key
                            for key, value in deal.items()
                            if value not in (None, "", [])
                        )
            else:
                raise ValueError(f"Unsupported task: {task}")

    missing_declared = sorted(expected - declared)
    unexpected_declared = sorted(declared - expected)
    missing_positive = sorted(expected - positive)
    # Structure rows carry the complete schema declaration. NER rows declare
    # only entity types present in that row, so their union is label coverage.
    contract_missing_declared = missing_declared if task == "structure" else []
    if contract_missing_declared or unexpected_declared:
        raise ValueError(
            f"{task} training field contract failed for {path}: "
            f"missing_declared={contract_missing_declared}, "
            f"unexpected_declared={unexpected_declared}"
        )
    if missing_positive and not allow_missing_labels:
        raise ValueError(
            f"{task} training data has no positive labels for fields "
            f"{missing_positive}. Add labeled data or use "
            "--allow-missing-field-labels only for a diagnostic smoke run."
        )
    return {
        "status": "passed",
        "schema_field_count": len(expected),
        "declared_field_count": len(declared),
        "positive_label_field_count": len(positive),
        "fields_without_positive_labels": missing_positive,
        "all_fields_have_positive_labels": not missing_positive,
        "diagnostic_missing_labels_allowed": bool(
            missing_positive and allow_missing_labels
        ),
    }
