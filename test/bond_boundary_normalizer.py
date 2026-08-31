"""Deterministic boundary cleanup for the bond-deal business schema.

The extractor is responsible for deciding which field a span belongs to.  This
module only removes business cue words that are routinely attached to an
otherwise correct span.  Keeping the rules here (rather than in GLiNER2's
generic decoder) prevents bond-specific conventions from affecting other
users of the library.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


SEND_TYPE_VALUES = (
    "固定收益平台",
    "大宗",
    "竞价",
    "请求",
    "对话",
    "固收",
)

_INSTITUTION_PREFIX = re.compile(
    r"^(?:固收联系|发请求给|请求发给|请求发|对话发|发给|走|发)"
)
_INSTITUTION_SUFFIX = re.compile(r"(?:请求|对话)$")
_DATE_SUFFIX = re.compile(r"(?:上交所固收|交易所|交易|交割|T)$")
_CALL_YIELD_SUFFIX = re.compile(r"\s*行权\s*$")
_CONTACT_PREFIX = re.compile(r"^(?:固收联系|联系)")
_FEE_PREFIX = re.compile(r"^留")
_TRADER_SUFFIX = re.compile(r"收对话$")


def _nonempty_or_original(cleaned: str, original: str) -> str:
    cleaned = cleaned.strip()
    return cleaned if cleaned else original


def normalize_boundary_text(field: str, text: str) -> str:
    """Return the business-canonical substring for one predicted field."""
    original = text.strip()
    cleaned = original

    if field in {"send_to", "bridge_institution"}:
        cleaned = _INSTITUTION_PREFIX.sub("", cleaned)
        cleaned = _INSTITUTION_SUFFIX.sub("", cleaned)
    elif field in {"send_type", "buyer_send_type", "seller_send_type"}:
        for value in SEND_TYPE_VALUES:
            if value in cleaned:
                return value
    elif field == "call_yield":
        cleaned = _CALL_YIELD_SUFFIX.sub("", cleaned)
    elif field == "settlement_date":
        cleaned = _DATE_SUFFIX.sub("", cleaned)
    elif field == "contact_institution":
        cleaned = _CONTACT_PREFIX.sub("", cleaned)
    elif field in {"fee", "buyer_fee", "seller_fee"}:
        cleaned = _FEE_PREFIX.sub("", cleaned)
    elif field in {"send_to_trader", "send_from_trader"}:
        # The cue must be attached to a non-trivial name; do not strip a
        # possible one-character surname from a two-character prediction.
        if len(cleaned) >= 3 and cleaned.startswith(("发", "给")):
            cleaned = cleaned[1:]
        cleaned = _TRADER_SUFFIX.sub("", cleaned)

    return _nonempty_or_original(cleaned, original)


def normalize_field_boundary(field: str, value: Any) -> Any:
    """Normalize strings and list-valued fields without changing their shape."""
    if isinstance(value, str):
        return normalize_boundary_text(field, value)
    if isinstance(value, tuple):
        return tuple(normalize_field_boundary(field, item) for item in value)
    if isinstance(value, list):
        return [normalize_field_boundary(field, item) for item in value]
    return value


def _normalize_prediction_value(field: str, value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_prediction_value(field, item) for item in value]
    if isinstance(value, dict):
        result = dict(value)
        key = "text" if "text" in result else "value" if "value" in result else None
        if key is None or not isinstance(result[key], str):
            return result
        original = result[key]
        normalized = normalize_boundary_text(field, original)
        result[key] = normalized
        if (
            normalized != original
            and "start" in result
            and "end" in result
            and normalized in original
        ):
            offset = original.find(normalized)
            result["start"] = int(result["start"]) + offset
            result["end"] = result["start"] + len(normalized)
        return result
    if isinstance(value, str):
        return normalize_boundary_text(field, value)
    return value


def normalize_prediction_boundaries(prediction: Any) -> Any:
    """Return a normalized copy of an official formatted prediction.

    Supported shapes are ``{"deal": [...]}``, training-style
    ``{"json_structures": [...]}``, and raw lists of deal dictionaries.
    Confidence metadata is retained; span offsets are tightened when present.
    """
    result = deepcopy(prediction)

    def normalize_deal(deal: Any) -> Any:
        if not isinstance(deal, dict):
            return deal
        return {
            field: _normalize_prediction_value(field, value)
            for field, value in deal.items()
        }

    if isinstance(result, list):
        return [
            {**item, "deal": normalize_deal(item["deal"])}
            if isinstance(item, dict) and isinstance(item.get("deal"), dict)
            else normalize_deal(item)
            for item in result
        ]
    if not isinstance(result, dict):
        return result
    if isinstance(result.get("deal"), list):
        result["deal"] = [normalize_deal(deal) for deal in result["deal"]]
    if isinstance(result.get("json_structures"), list):
        structures = []
        for item in result["json_structures"]:
            if isinstance(item, dict) and isinstance(item.get("deal"), dict):
                item = dict(item)
                item["deal"] = normalize_deal(item["deal"])
            structures.append(item)
        result["json_structures"] = structures
    return result
