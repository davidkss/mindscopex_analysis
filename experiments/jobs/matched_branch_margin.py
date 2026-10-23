"""Run only the pinned Fresh v2 first-divergence margin diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mindscopex_analysis.growth_control_inventory import load_inventory  # noqa: E402
from experiments.runners.config import job_name, load_toml, run_name, table  # noqa: E402

SCRIPT = ROOT / "scripts/diagnose_fresh_v2_matched_branch_margin.py"
CSV_NAME = "fresh_v2_f30908_matched_branch_margin.csv"
PINNED = {"layer": 5, "feature": 30908, "coefficients": [0.0, 7.0, 8.0], "only": "all"}


def run(config_path: Path, output_root: Path) -> Path:
    config = load_toml(config_path)
    if job_name(config) != "matched_branch_margin":
        raise ValueError("Expected run.job=matched_branch_margin")
    diagnostic = dict(table(config, "diagnostic"))
    selection = diagnostic.pop("case_ids_file", None)
    expected_hash = diagnostic.pop("case_ids_sha256", None)
    inventory = None
    if selection is None:
        if diagnostic != PINNED or expected_hash is not None:
            raise ValueError(f"diagnostic must match pinned settings: {PINNED}")
    else:
        if diagnostic != dict(PINNED, only="growth") or not isinstance(expected_hash, str):
            raise ValueError("Growth controls require pinned settings, only=growth and inventory SHA-256")
        selection = (ROOT / selection).resolve()
        if not selection.is_relative_to(ROOT.resolve()):
            raise ValueError("Inventory must be inside the source repository")
        inventory = load_inventory(selection, expected_hash)
        if not inventory["case_ids"]:
            raise ValueError("No additional Growth cases to measure")
    name = run_name(config)
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("run.name must be a single directory name")
    run_dir = output_root / name
    if run_dir.is_symlink():
        raise FileExistsError(f"Refusing symlink run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    # The bootstrap owns job.log. Reserve an exclusive subdirectory before any
    # job writes so direct runs also cannot replace previous diagnostic results.
    diagnostic_dir = run_dir / "diagnostics"
    diagnostic_dir.mkdir(exist_ok=False)
    shutil.copyfile(config_path, diagnostic_dir / "config.toml")
    if inventory is not None:
        shutil.copyfile(selection, diagnostic_dir / "inventory.json")
    output = diagnostic_dir / (
        "fresh_v2_growth_controls_f30908_branch_margin.csv" if inventory is not None else CSV_NAME)
    command = [
        sys.executable, "-u", str(SCRIPT), "--output", str(output.resolve()),
        "--layer", "5", "--feature", "30908", "--only", diagnostic["only"],
    ]
    if inventory is not None:
        command.extend(["--case-ids", str(selection)])
    manifest = {
        "job": "matched_branch_margin", "command": command,
        "model_id": "Qwen/Qwen3.5-2B-Base", "dataset": "crt_fresh_v2",
        "layer": 5, "feature": 30908, "coefficients": [0.0, 7.0, 8.0],
        "token_position": "all", "expected_rate_cases": 10, "expected_growth_cases": 6,
        "margin_definition": "lure_branch_logit - correct_branch_logit",
        "started_at": datetime.now(UTC).isoformat(), "status": "running",
    }
    if inventory is not None:
        manifest.update(expected_rate_cases=0, expected_growth_cases=len(inventory["case_ids"]),
                        case_ids=inventory["case_ids"],
                        inventory_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
                        source_inputs=inventory["inputs"])
    manifest_path = diagnostic_dir / "manifest.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save_manifest()
    print("+ " + " ".join(command), flush=True)
    try:
        # Inherit stdout/stderr: remote_bootstrap tees both into run_dir/job.log.
        subprocess.run(command, cwd=ROOT, check=True)
    except Exception:
        manifest["status"] = "failed"
        raise
    else:
        manifest["status"] = "succeeded"
    finally:
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        save_manifest()
    print(f"ARTIFACT_DIR={run_dir}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiments"))
    args = parser.parse_args()
    run(args.config.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
