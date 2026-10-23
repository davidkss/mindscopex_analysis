"""Colab Job adapter for the pinned 33-case full-answer likelihood diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from experiments.runners.config import job_name, load_toml, run_name, table
from mindscopex_analysis.full_answer_protocols import require_protocol
from mindscopex_analysis.growth_control_inventory import load_inventory

SCRIPT = ROOT / "scripts/diagnose_fresh_v2_full_answer_likelihood.py"
PINNED = dict(layer=5, feature=30908, coefficients=[0.0, 7.0, 8.0], token_position="all")


def run(config_path, output_root):
    config = load_toml(config_path)
    if job_name(config) != "full_answer_likelihood":
        raise ValueError("Expected run.job=full_answer_likelihood")
    cfg = dict(table(config, "diagnostic"))
    input_protocol = require_protocol(cfg.pop("input_protocol", None))
    selection = (ROOT / cfg.pop("inventory")).resolve()
    expected_hash = cfg.pop("inventory_sha256")
    include_eos = cfg.pop("include_eos")
    terminal = cfg.pop("eos_token_id", None)
    if cfg != PINNED or type(include_eos) is not bool:
        raise ValueError("Full-answer Job requires pinned conditions and boolean include_eos")
    if terminal is not None and (not include_eos or type(terminal) is not int):
        raise ValueError("eos_token_id requires include_eos=true and an integer")
    if not selection.is_relative_to(ROOT.resolve()):
        raise ValueError("Inventory must be inside the source repository")
    data = load_inventory(selection, expected_hash)
    if len(data["cases"]) != 33 or len(data["slots"]) != 64:
        raise ValueError("Expected exactly 33 Growth cases / 64 slots")
    name = run_name(config)
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("run.name must be a single directory name")
    run_dir = output_root / name
    if run_dir.is_symlink():
        raise FileExistsError("Refusing symlink run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    # Only the current bootstrap's log may predate this Job. Reject prior
    # artifacts before claiming or writing anything, including old manifests.
    if any(path.name != "job.log" or path.is_symlink() or not path.is_file()
           for path in run_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty full-answer run directory: {run_dir}")
    with (run_dir / ".full_answer_job_claim").open("x") as stream:
        stream.write("Reserved once; do not reuse this run directory.\n")
    with (run_dir / "config.toml").open("xb") as stream:
        stream.write(config_path.read_bytes())
    output = run_dir / "likelihood"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing diagnostic output: {output}")
    command = [sys.executable, "-u", str(SCRIPT), "--inventory", str(selection),
               "--output-dir", str(output), "--input-protocol", input_protocol]
    if include_eos:
        command.append("--include-eos")
    if terminal is not None:
        command.extend(["--eos-token-id", str(terminal)])
    metadata = dict(command=command, inventory_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
                    input_protocol=input_protocol, eos_policy="include" if include_eos else "exclude",
                    expected_score_rows=99, expected_candidate_paths=198, include_eos=include_eos,
                    status="running")
    manifest = run_dir / "job_manifest.json"
    with manifest.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")
    try:
        print("+ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    except Exception:
        metadata["status"] = "failed"
        raise
    else:
        metadata["status"] = "succeeded"
    finally:
        manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
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
