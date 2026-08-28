from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "test" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"bond_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_raw_data_and_business_schema_cover_same_76_fields():
    prepare = load_script("prepare_data")
    raw = ROOT / "test" / "data" / "bond_deal_0805_structured_aug_v1_sample_10000.jsonl"
    samples = [json.loads(line) for line in raw.open(encoding="utf-8")]
    contract = prepare.validate_full_schema(samples, prepare.load_descriptions())

    assert contract["status"] == "passed"
    assert contract["schema_field_count"] == 76
    assert contract["raw_field_count"] == 76
    assert contract["all_fields_have_non_null_labels"] is True


def test_evaluation_contract_rejects_a_partial_declared_schema():
    evaluate = load_script("evaluate_sentence_acc")
    descriptions = {"deal": {"bond_code": "code", "buyer": "buyer"}}
    rows = [
        {
            "output": {
                "json_descriptions": {"deal": {"bond_code": "code"}},
                "json_structures": [{"deal": {"bond_code": "123"}}],
            }
        }
    ]

    try:
        evaluate.validate_field_contract(rows, descriptions)
    except ValueError as exc:
        assert "missing_declared=['buyer']" in str(exc)
    else:
        raise AssertionError("partial schema should fail the evaluation contract")


def test_evaluation_contract_reports_fields_without_positive_support():
    evaluate = load_script("evaluate_sentence_acc")
    descriptions = {"deal": {"bond_code": "code", "buyer": "buyer"}}
    rows = [
        {
            "output": {
                "json_descriptions": descriptions,
                "json_structures": [
                    {"deal": {"bond_code": "123", "buyer": None}}
                ],
            }
        }
    ]

    contract = evaluate.validate_field_contract(rows, descriptions)
    assert contract["status"] == "passed"
    assert contract["fields_without_non_null_gold_support"] == ["buyer"]
    assert contract["all_fields_have_non_null_gold_support"] is False


def test_per_field_scores_count_wrong_value_as_fp_and_fn():
    evaluate = load_script("evaluate_sentence_acc")
    gold = [(('bond_code', '123'), ('buyer', 'A'))]
    pred = [(('bond_code', '456'), ('buyer', 'A'))]
    scores = evaluate.per_field_scores(gold, pred)

    assert scores["buyer"] == {"tp": 1}
    assert scores["bond_code"] == {"fp": 1, "fn": 1}


def test_training_contract_rejects_missing_field_labels(tmp_path):
    contract = load_script("field_contract")
    path = tmp_path / "ner.jsonl"
    path.write_text(
        json.dumps(
            {
                "output": {
                    "entities": {"bond_code": ["123"]},
                    "entity_descriptions": {"bond_code": "code"},
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        contract.validate_training_fields(path, "full", "ner")
    except ValueError as exc:
        assert "training data has no positive labels" in str(exc)
    else:
        raise AssertionError("incomplete full-schema training data should fail")
