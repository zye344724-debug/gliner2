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


def test_sample_cap_preserves_all_76_positive_fields():
    prepare = load_script("prepare_data")
    raw = ROOT / "test" / "data" / "bond_deal_0805_structured_aug_v1_sample_10000.jsonl"
    samples = [json.loads(line) for line in raw.open(encoding="utf-8")]
    fields = set(prepare.load_descriptions()["deal"])

    selected, stats = prepare.select_coverage_balanced_samples(
        samples, max_samples=1000, fields=fields, seed=42
    )
    contract = prepare.validate_full_schema(selected, prepare.load_descriptions())

    assert len(selected) == 1000
    assert stats["enabled"] is True
    assert stats["source_samples"] == 10000
    assert stats["selected_samples"] == 1000
    assert set(stats["selected_positive_counts"]) == fields
    assert min(stats["selected_positive_counts"].values()) >= 1
    assert contract["schema_field_count"] == 76
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


def test_per_field_threshold_tuning_uses_cached_confidences():
    evaluate = load_script("evaluate_sentence_acc")
    rows = [
        {
            "output": {
                "json_structures": [
                    {"deal": {"bond_code": "A", "buyer": "机构甲"}}
                ]
            }
        },
        {
            "output": {
                "json_structures": [{"deal": {"bond_code": "B"}}]
            }
        },
    ]
    predictions = [
        {
            "deal": [
                {
                    "bond_code": {"text": "A", "confidence": 0.4},
                    "buyer": {"text": "机构甲", "confidence": 0.9},
                }
            ]
        },
        {
            "deal": [
                {
                    "bond_code": {"text": "B", "confidence": 0.4},
                    "buyer": {"text": "错误机构", "confidence": 0.2},
                }
            ]
        },
    ]

    thresholds, diagnostics = evaluate.tune_field_thresholds(
        rows,
        predictions,
        ["bond_code", "buyer"],
        [0.3, 0.5, 0.7],
        0.5,
    )
    filtered = [
        evaluate.filter_prediction_by_field_thresholds(pred, thresholds, 0.5)
        for pred in predictions
    ]

    assert thresholds["bond_code"] == 0.3
    assert evaluate.exact_accuracy(rows, filtered) == 1.0
    assert diagnostics["exact_after_coordinate_tuning"] == 1.0


def test_bond_boundary_normalization_is_prediction_only():
    evaluate = load_script("evaluate_sentence_acc")
    gold = {
        "json_structures": [
            {
                "deal": {
                    "send_to": ["南京银行"],
                    "bridge_institution": "南京银行",
                    "send_type": "请求",
                    "call_yield": "1.79",
                    "settlement_date": "07.07",
                }
            }
        ]
    }
    prediction = {
        "deal": [
            {
                "send_to": ["发南京银行"],
                "bridge_institution": "发请求给南京银行",
                "send_type": "发南京银行请求",
                "call_yield": "1.79行权",
                "settlement_date": "07.07交易所",
            }
        ]
    }

    gold_deals = evaluate.deals_from_gold_output(gold)
    raw_pred = evaluate.deals_from_prediction(
        prediction, normalize_boundaries=False
    )
    normalized_pred = evaluate.deals_from_prediction(prediction)

    assert evaluate.sentence_exact_match(gold_deals, raw_pred) is False
    assert evaluate.sentence_exact_match(gold_deals, normalized_pred) is True


def test_boundary_normalizer_keeps_nonempty_and_unrelated_values():
    normalizer = load_script("bond_boundary_normalizer")

    assert normalizer.normalize_boundary_text("send_to", "发") == "发"
    assert normalizer.normalize_boundary_text("buyer", "发改委") == "发改委"
    assert normalizer.normalize_boundary_text("send_to_trader", "发财") == "发财"
    assert normalizer.normalize_boundary_text("seller_fee", "留4厘") == "4厘"


def test_boundary_normalizer_preserves_confidence_and_tightens_offsets():
    normalizer = load_script("bond_boundary_normalizer")
    prediction = {
        "deal": [
            {
                "send_to": {
                    "text": "发南京银行",
                    "confidence": 0.9,
                    "start": 10,
                    "end": 15,
                }
            }
        ]
    }

    normalized = normalizer.normalize_prediction_boundaries(prediction)

    assert prediction["deal"][0]["send_to"]["text"] == "发南京银行"
    assert normalized["deal"][0]["send_to"] == {
        "text": "南京银行",
        "confidence": 0.9,
        "start": 11,
        "end": 15,
    }


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
