"""Colab adapter for one Growth019 coefficient (0, 7, or 8) per run."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runners.config import job_name, load_toml, run_name, table
from experiments.trace_fresh_v2_first_divergence import INPUTS, filter_plan, load_plan, write_new

PINNED = dict(case_id="crt_fresh_v2_growth_019", layer=5,
              feature=30908, check_equivalence=True)
ALLOWED_COEFFICIENTS = (0, 7, 8)


def prepare_plan(config):
    cfg = table(config, "diagnostic")
    if (job_name(config) != "first_divergence_trace" or {k: v for k, v in cfg.items() if k != "coefficient"} != PINNED
            or type(cfg.get("coefficient")) is not int
            or cfg.get("coefficient") not in ALLOWED_COEFFICIENTS
            or cfg.get("check_equivalence") is not True):
        raise ValueError("First trace requires Growth019, coefficient 0/7/8, Layer 5/30908 and equivalence=true")
    name = run_name(config)
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("run.name must be a single directory name")
    plan = filter_plan(load_plan(), case_id=cfg["case_id"], coefficient=cfg["coefficient"],
                       check_equivalence=cfg["check_equivalence"])
    print(json.dumps(dict(case_ids=plan["case_ids"], coefficients=plan["coefficients"],
                         n_unique_measurements=plan["n_unique_measurements"],
                         seeds=[s["seed"] for s in plan["slot_comparisons"]],
                         n_slot_comparisons=plan["n_slot_comparisons"],
                         expected_generate_calls=plan["expected_generate_calls"])), flush=True)
    return plan


def include_reference_inputs(root, archive_path):
    """Existing source packer excludes results; include only the three required CSVs."""
    with zipfile.ZipFile(archive_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        existing = set(archive.namelist())
        for source in INPUTS.values():
            relative = source.relative_to(ROOT)
            path = root / relative
            name = relative.as_posix()
            if name in existing:
                if archive.read(name) != path.read_bytes():
                    raise ValueError(f"Archive input differs: {name}")
            else:
                archive.write(path, name)


def validate_equivalence(output, plan):
    """Require affirmative evidence per selected measurement; never infer missing checks."""
    measurements = output.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != len(plan["measurements"]):
        raise ValueError("Equivalence validation failed: selected measurements missing or count differs")
    for expected, measurement in zip(plan["measurements"], measurements, strict=True):
        if not isinstance(measurement, dict) or measurement.get("measurement_id") != expected["measurement_id"]:
            raise ValueError("Equivalence validation failed: measurement identity differs")
        capture = measurement.get("capture")
        check = capture.get("equivalence_check") if isinstance(capture, dict) else None
        if not isinstance(check, dict) or check.get("status") != "compared":
            raise ValueError(f"Equivalence validation failed: {expected['measurement_id']} status is not compared")
        for field in ("generated_token_ids_equal", "answers_equal", "equivalent"):
            if check.get(field) is not True:
                raise ValueError(f"Equivalence validation failed: {expected['measurement_id']} {field} is not True")
    # Reuse preflight hashes; do not introduce another digest implementation.
    sources = output.get("source_files")
    if not isinstance(sources, dict) or sources.keys() != plan["source_files"].keys():
        raise ValueError("Input SHA-256 metadata missing or input identities differ")
    for name, source in plan["source_files"].items():
        record = sources[name]
        if not isinstance(record, dict) or record.get("sha256") != source["sha256"]:
            raise ValueError(f"Input SHA-256 metadata differs: {name}")


def run(config_path, output_root):
    config = load_toml(config_path)
    plan = prepare_plan(config)
    run_dir = output_root / run_name(config)
    if run_dir.is_symlink():
        raise FileExistsError("Refusing symlink run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(p.name != "job.log" or p.is_symlink() or not p.is_file() for p in run_dir.iterdir()):
        raise FileExistsError(f"Refusing existing trace artifacts: {run_dir}")
    with (run_dir / ".trace_job_claim").open("x"):
        pass
    with (run_dir / "config.toml").open("xb") as stream:
        stream.write(config_path.read_bytes())
    write_new(run_dir / "plan.json", plan)
    command = [sys.executable, "-u", str(ROOT / "experiments/trace_fresh_v2_first_divergence.py"),
               "--run-model", "--case-id", config["diagnostic"]["case_id"],
               "--coefficient", str(config["diagnostic"]["coefficient"]),
               "--check-equivalence", "--output", str(run_dir / "actual_first_divergence.json")]
    metadata = dict(status="running", command=command, expected_generate_calls=2,
                    source_files=plan["source_files"])
    manifest = run_dir / "job_manifest.json"
    write_new(manifest, metadata)
    try:
        with (run_dir / "diagnostic.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        output = json.loads((run_dir / "actual_first_divergence.json").read_text())
        if (output.get("status") != "captured" or output.get("n_unique_measurements") != 1
                or output.get("n_slot_comparisons") != 2 or output.get("expected_generate_calls") != 2
                or output.get("filters") != dict(case_id=PINNED["case_id"],
                                                       coefficient=config["diagnostic"]["coefficient"])):
            raise ValueError("Remote diagnostic did not produce the selected captured result")
        validate_equivalence(output, plan)
        metadata.update(status="succeeded", equivalence_validation="passed")
    except Exception as exc:
        metadata.update(status="failed", error=str(exc), returncode=getattr(exc, "returncode", None))
        message = f"TRACE_JOB_FAILED: {exc}\n"
        print(message, end="", flush=True)  # Captured by the existing bootstrap job.log.
        try:
            with (run_dir / "diagnostic.log").open("a", encoding="utf-8") as log:
                log.write(message)
        except OSError as log_error:
            print(f"Could not append diagnostic failure log: {log_error}", flush=True)
        raise
    finally:
        manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"ARTIFACT_DIR={run_dir}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
