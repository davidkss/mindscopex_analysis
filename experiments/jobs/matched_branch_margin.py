"""Run only the pinned Fresh v2 first-divergence margin diagnostic."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runners.config import job_name, load_toml, run_name, table  # noqa: E402

SCRIPT = ROOT / "scripts/diagnose_fresh_v2_matched_branch_margin.py"
CSV_NAME = "fresh_v2_f30908_matched_branch_margin.csv"
PINNED = {"layer": 5, "feature": 30908, "coefficients": [0.0, 7.0, 8.0], "only": "all"}


def run(config_path: Path, output_root: Path) -> Path:
    config = load_toml(config_path)
    if job_name(config) != "matched_branch_margin":
        raise ValueError("Expected run.job=matched_branch_margin")
    if table(config, "diagnostic") != PINNED:
        raise ValueError(f"diagnostic must match pinned settings: {PINNED}")
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
    output = diagnostic_dir / CSV_NAME
    command = [
        sys.executable, "-u", str(SCRIPT), "--output", str(output.resolve()),
        "--layer", "5", "--feature", "30908", "--only", "all",
    ]
    manifest = {
        "job": "matched_branch_margin", "command": command,
        "model_id": "Qwen/Qwen3.5-2B-Base", "dataset": "crt_fresh_v2",
        "layer": 5, "feature": 30908, "coefficients": [0.0, 7.0, 8.0],
        "token_position": "all", "expected_rate_cases": 10, "expected_growth_cases": 6,
        "margin_definition": "lure_branch_logit - correct_branch_logit",
        "started_at": datetime.now(UTC).isoformat(), "status": "running",
    }
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
