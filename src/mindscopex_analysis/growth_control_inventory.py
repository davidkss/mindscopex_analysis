"""Validate the portable Growth control inventory shared by CPU and Colab jobs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CONDITIONS = {
    "model_id": "Qwen/Qwen3.5-2B-Base", "dataset": "crt_fresh_v2",
    "layer": 5, "feature": 30908, "coefficients": [0.0, 7.0, 8.0],
    "token_position": "all", "margin": "lure_branch_logit - correct_branch_logit",
}


def load_inventory(path: Path, expected_sha256: str | None = None) -> dict:
    raw = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Inventory SHA-256 mismatch: {path}")
    data = json.loads(raw)
    if data.get("schema") != 1 or data.get("conditions") != CONDITIONS:
        raise ValueError(f"Unsupported Growth inventory conditions: {path}")
    cases, slots, selected, existing = (
        data["cases"], data["slots"], data["case_ids"], data["existing_margins"])
    if not cases or not slots or not isinstance(selected, list):
        raise ValueError("Inventory must contain cases, slots and a case_ids list")
    ids = [row["case_id"] for row in cases]
    if len(set(ids)) != len(ids) or any(not cid.startswith("crt_fresh_v2_growth_") for cid in ids):
        raise ValueError("Inventory has duplicate/non-Growth case IDs")
    keys = [(row["seed"], row["case_id"]) for row in slots]
    if len(set(keys)) != len(keys) or {row["case_id"] for row in slots} != set(ids):
        raise ValueError("Inventory slot duplicates or case coverage mismatch")
    if any(row["plus7_label"] not in {"correct", "lure"}
           or row["plus8_label"] not in {"correct", "lure"} for row in slots):
        raise ValueError("Invalid inventory behavioral labels")
    old_ids = [row["case_id"] for row in existing]
    if len(set(old_ids)) != len(old_ids) or not set(old_ids) <= set(ids):
        raise ValueError("Inventory has duplicate/unexpected existing margin cases")
    if len(set(selected)) != len(selected) or set(selected) != set(ids) - set(old_ids):
        raise ValueError("case_ids must be exactly the unmeasured Growth cases")
    for row in cases:
        cid = row["case_id"]
        case_slots = [s for s in slots if s["case_id"] == cid]
        groups = sorted({s["plus7_label"] + "_to_" + s["plus8_label"] for s in case_slots})
        if (row["evaluation_slots"] != len(case_slots) or row["groups"] != groups
                or row["needs_measurement"] != (cid in selected)):
            raise ValueError(f"Inventory case aggregation differs from slots: {cid}")
    return data
