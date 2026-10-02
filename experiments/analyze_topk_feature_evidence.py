"""Read-only Top-K evidence inventory. Stdlib only; never run or load a model.

Select explicit protocol groups, never an older successful run as a substitute
for a newer invalid run. All repeated runs are inventoried before selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics as st
import tempfile
import tomllib
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = (30908, 22552, 14817, 20108, 30475)
NA = "not available"
FAMILIES = ("crt_difference", "crt_rate", "crt_growth")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(a, b, context):
    require(
        math.isfinite(float(a))
        and math.isfinite(float(b))
        and math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12),
        context,
    )


def read_json(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index(rows, key="case_id"):
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows) and bool(result), f"Empty/duplicate {key}")
    return result


def behavioral_stats(before, after):
    n = len(before)
    ba = sum(r["label"] == "correct" for r in before.values()) / n
    sa = sum(r["label"] == "correct" for r in after.values()) / n
    bl = sum(r["label"] == "lure" for r in before.values()) / n
    sl = sum(r["label"] == "lure" for r in after.values()) / n
    return dict(
        n_cases=n,
        baseline_accuracy=ba,
        steered_accuracy=sa,
        accuracy_delta=sa - ba,
        baseline_lure_rate=bl,
        steered_lure_rate=sl,
        lure_rate_delta=sl - bl,
        fixes=sum(before[k]["label"] == "lure" and after[k]["label"] == "correct" for k in before),
        regressions=sum(
            before[k]["label"] == "correct" and after[k]["label"] == "lure" for k in before
        ),
    )


def behavioral(path, config, feature):
    recorded = {float(r["coefficient"]): r for r in csv_rows(path / "behavioral.csv")}
    detail = read_json(path / "behavioral/generations.json")
    coefficients = [float(r["coefficient"]) for r in detail]
    require(
        len(recorded)
        == len(csv_rows(path / "behavioral.csv"))
        == len(set(coefficients))
        == len(detail),
        "Duplicate behavioral coefficients",
    )
    require(
        set(coefficients) == set(recorded) == set(config["behavioral"]["coefficients"]),
        "Behavioral coefficient mismatch",
    )
    result = []
    for entry in detail:
        require(entry["output_mode"] == config["behavioral"]["output_mode"], "Output mode mismatch")
        if "feature_ids" in feature:
            require(
                entry.get("feature_ids") == feature["feature_ids"]
                and entry.get("layer") == feature["layer"],
                "Combined generation feature mismatch",
            )
        before, after = index(entry["baseline_rows"]), index(entry["steered_rows"])
        require(set(before) == set(after), "Baseline/steered case set mismatch")
        for cid in before:
            require(before[cid]["family"] == after[cid]["family"], "Behavioral family mismatch")
            allowed = (
                ("correct", "lure")
                if entry["output_mode"] == "binary_choice"
                else ("correct", "lure", "both", "other")
            )
            require(
                before[cid]["label"] in allowed and after[cid]["label"] in allowed, "Invalid label"
            )
        metrics = behavioral_stats(before, after)
        for key in (
            "baseline_accuracy",
            "steered_accuracy",
            "accuracy_delta",
            "baseline_lure_rate",
            "steered_lure_rate",
            "lure_rate_delta",
        ):
            close(metrics[key], recorded[entry["coefficient"]][key], f"CSV reconstruction: {key}")
        families = []
        for family in sorted({r["family"] for r in before.values()}):
            b = {k: r for k, r in before.items() if r["family"] == family}
            families.append(dict(family=family, **behavioral_stats(b, {k: after[k] for k in b})))
        result.append(dict(coefficient=entry["coefficient"], **metrics, families=families))
    return result


def margins(path, kind, feature):
    rows = csv_rows(path / f"{kind}.csv")
    index(rows)
    summary = read_json(path / kind / "summary.json")
    require(summary["n_cases"] == len(rows), f"{kind} N mismatch")
    if kind == "causal_heldout":
        require(
            summary["feature_id"] == feature["feature_id"] and summary["layer"] == feature["layer"],
            "Causal feature mismatch",
        )
        for row in rows:
            close(
                float(row["baseline_margin"]) - float(row["edited_margin"]),
                row["margin_delta"],
                "Causal delta mismatch",
            )

        def aggregate(items):
            delta = [float(r["margin_delta"]) for r in items]
            return dict(
                n_cases=len(items),
                baseline_mean_margin=st.mean(float(r["baseline_margin"]) for r in items),
                intervention_mean_margin=st.mean(float(r["edited_margin"]) for r in items),
                mean_margin_delta=st.mean(delta),
                std_margin_delta=st.pstdev(delta),
                frac_positive=sum(x > 0 for x in delta) / len(delta),
            )

        keys = ("mean_margin_delta", "std_margin_delta", "frac_positive")
    else:
        for row in rows:
            close(
                float(row["hostile_margin_delta"]) - float(row["control_margin_delta"]),
                row["specificity_gap"],
                "Specificity gap mismatch",
            )

        def aggregate(items):
            return dict(
                n_cases=len(items),
                **{
                    f"mean_{key}": st.mean(float(r[column]) for r in items)
                    for key, column in (
                        ("hostile_delta", "hostile_margin_delta"),
                        ("control_delta", "control_margin_delta"),
                        ("specificity_gap", "specificity_gap"),
                    )
                },
            )

        keys = ("mean_hostile_delta", "mean_control_delta", "mean_specificity_gap")
    result = aggregate(rows)
    for key in keys:
        close(result[key], summary[key], f"{kind} summary mismatch {key}")
    result["families"] = [
        dict(family=f, **aggregate([r for r in rows if r["family"] == f]))
        for f in sorted({r["family"] for r in rows})
    ]
    result["validation"] = (
        "per-case baseline-minus-edited and summary reconstructed; original logits unavailable"
        if kind == "causal_heldout"
        else (
            "per-case gap and summary reconstructed; original control baseline"
            "/edited margins not saved"
        )
    )
    return result


def discovery(path):
    rows = csv_rows(path / "discover_features.csv")
    require(
        len({(r["layer"], r["feature_id"]) for r in rows}) == len(rows),
        "Duplicate discovery candidates",
    )
    reports = (
        read_json(path / "discover/null_reports.json")
        if (path / "discover/null_reports.json").exists()
        else []
    )
    by_feature = {(int(r["layer"]), int(r["feature_id"])): r for r in rows}
    for report in reports:
        row = by_feature[report["layer"], report["feature_id"]]
        panel = report["panel"]
        index(panel)
        require(len(panel) == report["panel_n"], "Null panel N mismatch")
        close(
            st.mean(r["observed_delta"] for r in panel),
            report["observed_mean_delta"],
            "Null panel mean mismatch",
        )
        for prefix in ("gaussian", "peer"):
            values = report[f"{prefix}_values"]
            key = "gaussian_percentile" if prefix == "gaussian" else "peer_feature_percentile"
            close(
                sum(x < report["observed_mean_delta"] for x in values) / len(values),
                report[key],
                f"{key} mismatch",
            )
            close(st.mean(values), report[f"{prefix}_mean"], f"{prefix} mean mismatch")
        for key in (
            "observed_mean_delta",
            "gaussian_percentile",
            "peer_feature_percentile",
            "selection_adjusted_percentile",
            "selection_adjusted_p",
        ):
            if row.get(key) not in ("", None):
                close(row[key], report[key], f"Discovery CSV/null mismatch: {key}")
        close(
            report["selection_adjusted_percentile"] + report["selection_adjusted_p"],
            1,
            "Adjusted null complement",
        )
    sorted_rows = sorted(rows, key=lambda r: -float(r["mean_margin_delta"]))
    candidates = []
    for r in sorted_rows:
        if int(r["feature_id"]) in FEATURES:
            candidates.append(
                dict(
                    feature_id=int(r["feature_id"]),
                    layer=int(r["layer"]),
                    rank_by_mean_margin_delta=1
                    + sum(
                        float(x["mean_margin_delta"]) > float(r["mean_margin_delta"]) for x in rows
                    ),
                    metrics={
                        k: float(v) if v else NA
                        for k, v in r.items()
                        if k not in ("feature_id", "layer")
                    },
                )
            )
    return dict(
        candidates=candidates,
        total_candidates=len(rows),
        null_reports=reports,
        selected_feature=read_json(path / "study_feature.json")["feature"],
        validation=(
            "CSV/null panel mean and empirical percentiles checked; discovery "
            "per-case screening deltas and bootstrap maxima not saved"
        ),
    )


def inspect_run(path):
    config = tomllib.loads((path / "config.toml").read_text())
    art = path / "artifacts"
    manifest = read_json(art / "manifest.json") if (art / "manifest.json").exists() else {}
    meta = read_json(art / "study_feature.json") if (art / "study_feature.json").exists() else {}
    feature = config.get("features", config.get("feature", meta.get("feature", {})))
    ids = feature.get("feature_ids", [feature["feature_id"]] if "feature_id" in feature else [])
    kind = config["experiment"]["kind"]
    candidate_ids = []
    if (art / "discover_features.csv").exists():
        candidate_ids = [int(r["feature_id"]) for r in csv_rows(art / "discover_features.csv")]
    related = bool(set(ids + candidate_ids) & set(FEATURES)) or kind == "discover"
    files = sorted(str(p.relative_to(art)) for p in art.rglob("*") if p.is_file())
    log = (art / "job.log").read_text() if (art / "job.log").exists() else ""
    row = dict(
        run_directory=str(path.resolve()),
        run_name=config["run"]["name"],
        experiment_kind=kind,
        target_related=related,
        feature_ids=ids,
        discovery_target_candidates=sorted(set(candidate_ids) & set(FEATURES)),
        layer=feature.get("layer", config.get("discover", {}).get("layers", NA)),
        coefficient=config.get("behavioral", {}).get(
            "coefficients",
            feature.get("coefficient", config.get("discover", {}).get("coefficient", NA)),
        ),
        seed=config.get("data", {}).get("split_seed", NA),
        dataset=config.get("data", {}).get("dataset", NA),
        train_test_settings=config.get("data", {}),
        n_train=manifest.get("n_train", NA),
        n_test=manifest.get("n_test", NA),
        output_mode=config.get("behavioral", {}).get("output_mode", NA),
        model_id=manifest.get("model_id", NA),
        model_config=config.get("model", {}),
        artifact_files=files,
        manifest_finished=bool(manifest.get("finished_at")),
        status="INVALID",
        execution_outcome="not available",
        validation_errors=[],
        selected_for=[],
        selection_reason="",
        environment={
            k: manifest.get(k, NA)
            for k in (
                "python",
                "platform",
                "cuda_device",
                "sae_repo_id",
                "started_at",
                "finished_at",
            )
        },
        fallback_warning_count=log.count("falling back to its reference"),
        artifact_sha256={
            n: digest(art / n) for n in files if n.endswith((".csv", ".json", ".toml"))
        },
    )
    row["source_archive_sha256"] = (
        digest(path / "source.zip") if (path / "source.zip").exists() else NA
    )
    row["archived_code_sha256"] = {}
    if (path / "source.zip").exists():
        with zipfile.ZipFile(path / "source.zip") as archive:
            for name in (
                "src/mindscopex_analysis/research.py",
                "src/mindscopex_analysis/effects.py",
                "experiments/jobs/research_experiments.py",
            ):
                if name in archive.namelist():
                    row["archived_code_sha256"][name] = hashlib.sha256(
                        archive.read(name)
                    ).hexdigest()
    details = {}
    try:
        require(
            bool(manifest) and bool(manifest.get("finished_at")),
            "Missing finished manifest; no completed result available",
        )
        require(
            "JOB_FAILED" not in log and "Traceback (most recent call last)" not in log,
            "Failed job log",
        )
        require(
            tomllib.loads((art / "config.toml").read_text()) == config,
            "Launch/artifact config mismatch",
        )
        for key in ("dataset", "split_seed", "train_frac"):
            require(manifest[key] == config["data"][key], f"Manifest {key} mismatch")
        require(manifest["run_name"] == config["run"]["name"], "Manifest run name mismatch")
        for key in ("layer", "feature_id", "feature_ids"):
            if key in feature:
                require(meta["feature"][key] == feature[key], f"Pinned feature {key} mismatch")
        if "feature" in config or "features" in config:
            require(meta["source"] == "pinned", "Expected pinned feature")
        require(
            manifest.get("study_feature") == meta.get("feature"), "Manifest/study feature mismatch"
        )
        require(
            manifest.get("feature_source") == meta.get("source"), "Manifest feature source mismatch"
        )
        if (art / "behavioral.csv").exists():
            details["behavioral"] = behavioral(art, config, meta["feature"])
        for stage in ("causal_heldout", "control_specificity"):
            if (art / f"{stage}.csv").exists():
                details[stage] = margins(art, stage, meta["feature"])
        if (art / "discover_features.csv").exists():
            details["discover"] = discovery(art)
        stages = (
            ("discover", "causal_heldout", "control_specificity", "behavioral")
            if kind == "study"
            else (kind,)
        )
        require(all(k in details for k in stages), "Required stage artifact missing")
        if kind in ("causal_heldout", "control_specificity"):
            require(details[kind]["n_cases"] == manifest["n_test"], "Heldout evaluation N mismatch")
        row["status"] = "COMPLETED"
        row["execution_outcome"] = "completed manifest and validated artifacts"
    except (ValueError, KeyError, TypeError, OSError, ZeroDivisionError) as exc:
        row["validation_errors"].append(str(exc))
        row["execution_outcome"] = (
            "failed"
            if "JOB_FAILED" in log or "Traceback (most recent call last)" in log
            else "incomplete/unverified; not evidence of success"
        )
    return row, details, config


def analyze(root):
    inspected = [inspect_run(p) for p in sorted(root.iterdir()) if p.is_dir()]
    inventory = [x[0] for x in inspected]
    bypath = {r["run_directory"]: (r, d, c) for r, d, c in inspected}
    groups = defaultdict(list)
    for item in inspected:
        groups[item[0]["run_name"]].append(item)
    decisions = []
    discrepancies = []

    def select(name, purpose):
        items = groups.get(name, [])
        if not items:
            decisions.append(
                dict(protocol=name, purpose=purpose, selected=NA, reason="run artifact not found")
            )
            return None
        newest = max(items, key=lambda x: Path(x[0]["run_directory"]).name)
        valid = newest[0]["status"] == "COMPLETED"
        reason = (
            (
                "All attempts inspected; most recent attempt is complete and valid"
                "ates for this explicit protocol."
            )
            if valid
            else "Newest attempt is incomplete/invalid; no fallback to earlier successful runs."
        )
        decisions.append(
            dict(
                protocol=name,
                purpose=purpose,
                selected=newest[0]["run_directory"] if valid else NA,
                reason=reason,
                attempts=[dict(run=x[0]["run_directory"], status=x[0]["status"]) for x in items],
            )
        )
        if valid:
            newest[0]["selected_for"].append(purpose)
            newest[0]["selection_reason"] = reason
            return newest
        return None

    discovery_runs = [
        select(n, "discovery evidence: separate protocol, no cross-protocol replacement")
        for n in (
            "study_discover_topk_2b",
            "study_discover_topk_strongnull_2b",
            "study_discover_topk_strongnull_p18_2b",
        )
    ]
    anchor = discovery_runs[0]
    primary_path = ROOT / "results/analysis/combined_multiseed_primary_summary.json"
    primary = read_json(primary_path) if primary_path.exists() else None
    tables = []
    causal = []
    specificity = []
    behaviors = []
    coverage = []
    robustness = []
    for fid in FEATURES:
        c = select(f"study_causal_heldout_f{fid}_2b", "dedicated heldout")
        sp = select(f"study_control_specificity_f{fid}_2b", "dedicated specificity")
        b = select(
            f"study_behavioral_topk_f{fid}_2b",
            "common five-feature dose grid; +8 reference, not best-dose selection",
        )
        for chosen, kind, out in (
            (c, "causal_heldout", causal),
            (sp, "control_specificity", specificity),
        ):
            if chosen:
                require(
                    chosen[0]["feature_ids"] == [fid] and chosen[0]["layer"] == 5,
                    "Selected feature/layer mismatch",
                )
                require(
                    chosen[2]["feature"]["coefficient"] == 1.0
                    and chosen[2]["feature"]["intervention_mode"] == "remove_activation",
                    "Margin intervention config mismatch",
                )
                out.append(
                    dict(
                        feature_id=fid,
                        run=chosen[0]["run_directory"],
                        coefficient=1.0,
                        intervention_mode="remove_activation",
                        **chosen[1][kind],
                    )
                )
        if c and sp:
            causal_rows = index(
                csv_rows(Path(c[0]["run_directory"]) / "artifacts/causal_heldout.csv")
            )
            specificity_rows = index(
                csv_rows(Path(sp[0]["run_directory"]) / "artifacts/control_specificity.csv")
            )
            require(set(causal_rows) == set(specificity_rows), "Causal/specificity case mismatch")
            for cid in causal_rows:
                close(
                    causal_rows[cid]["margin_delta"],
                    specificity_rows[cid]["hostile_margin_delta"],
                    "Cross-stage hostile discrepancy",
                )
        ref = None
        if b:
            for entry in b[1]["behavioral"]:
                behaviors.append(
                    dict(feature_id=fid, run=b[0]["run_directory"], seed=b[0]["seed"], **entry)
                )
            ref = next((r for r in b[1]["behavioral"] if r["coefficient"] == 8.0), None)
        multiseed = []
        if primary:
            condition = f"feature{fid}"
            for seed, paths in primary["run_provenance"].items():
                if condition not in paths:
                    continue
                run = paths[condition]
                item = bypath.get(run)
                if item is None or item[0]["status"] != "COMPLETED":
                    discrepancies.append(
                        dict(
                            feature_id=fid,
                            seed=seed,
                            reason="Prior primary run absent or invalid",
                            run=run,
                        )
                    )
                    continue
                item[0]["selected_for"].append(
                    "existing finalized multiseed primary provenance; +7"
                )
                item[0]["selection_reason"] = (
                    "Preserve finalized primary run provenance; independently reconstr"
                    "ucted, no replacement."
                )
                metric = next((x for x in item[1]["behavioral"] if x["coefficient"] == 7.0), None)
                require(metric is not None, "Prior primary +7 missing")
                prior = next(
                    x
                    for x in primary["analyses"]["primary"]["seed_summary"]
                    if x["seed"] == int(seed)
                )
                for key, suffix in (
                    ("baseline_accuracy", "baseline_accuracy"),
                    ("steered_accuracy", "accuracy"),
                    ("accuracy_delta", "accuracy_delta"),
                    ("fixes", "fixes"),
                    ("regressions", "regressions"),
                ):
                    if not math.isclose(
                        metric[key], prior[f"{condition}_{suffix}"], rel_tol=1e-12, abs_tol=1e-12
                    ):
                        discrepancies.append(
                            dict(
                                feature_id=fid,
                                seed=seed,
                                metric=key,
                                recomputed=metric[key],
                                prior=prior[f"{condition}_{suffix}"],
                            )
                        )
                multiseed.append(dict(seed=int(seed), run=run, **metric))
        observed_seeds = sorted(
            {
                r["seed"]
                for r, d, config in inspected
                if r["status"] == "COMPLETED" and r["feature_ids"] == [fid] and "behavioral" in d
            }
        )
        robustness.append(
            dict(
                feature_id=fid,
                completed_single_behavioral_seeds=observed_seeds,
                comparable_plus7_seed_count=len(multiseed),
                plus7_runs=multiseed,
                validation_note="No multi-seed claim from one seed or different coefficients",
            )
        )
        discovery_row = (
            next((r for r in anchor[1]["discover"]["candidates"] if r["feature_id"] == fid), None)
            if anchor
            else None
        )
        combined = []
        if primary:
            for seed, paths in primary["run_provenance"].items():
                item = bypath.get(paths["combined"])
                if item and fid in item[0]["feature_ids"] and item[0]["status"] == "COMPLETED":
                    combined.append(paths["combined"])
                    if "finalized combined provenance" not in item[0]["selected_for"]:
                        item[0]["selected_for"].append("finalized combined provenance")

        def state(name, selected):
            if selected:
                return "COMPLETED"
            if groups.get(name):
                return "INVALID"
            return (
                "CONFIG_ONLY"
                if (ROOT / "experiments/configs" / f"{name}.toml").exists()
                else "MISSING"
            )

        robust_state = (
            "COMPLETED"
            if len(multiseed) > 1
            else (
                "CONFIG_ONLY"
                if list(
                    (ROOT / "experiments/configs").glob(
                        f"study_behavioral_robust_f{fid}_seed*_2b.toml"
                    )
                )
                else "MISSING"
            )
        )
        states = dict(
            Discovery="COMPLETED" if discovery_row else "MISSING",
            Causal_heldout=state(f"study_causal_heldout_f{fid}_2b", c),
            Specificity=state(f"study_control_specificity_f{fid}_2b", sp),
            Behavioral=state(f"study_behavioral_topk_f{fid}_2b", b),
            Multi_seed_robustness=robust_state,
            Combined="COMPLETED" if combined else "MISSING",
        )
        coverage.append(dict(feature_id=fid, **states))
        tables.append(
            dict(
                feature_id=fid,
                layer=5,
                discovery_rank=discovery_row["rank_by_mean_margin_delta"] if discovery_row else NA,
                discovery_metric=(
                    "mean_margin_delta (baseline minus edited lure-correct logprob margin)"
                ),
                discovery_score=discovery_row["metrics"]["mean_margin_delta"]
                if discovery_row
                else NA,
                discovery_run=anchor[0]["run_directory"] if discovery_row else NA,
                causal_heldout_available=bool(c),
                causal_heldout_result=c[1]["causal_heldout"]["mean_margin_delta"] if c else NA,
                causal_run=c[0]["run_directory"] if c else NA,
                specificity_available=bool(sp),
                specificity_result=sp[1]["control_specificity"]["mean_specificity_gap"]
                if sp
                else NA,
                specificity_run=sp[0]["run_directory"] if sp else NA,
                behavioral_available=bool(b),
                behavioral_reference_coefficient=8.0 if ref else NA,
                behavioral_seed=b[0]["seed"] if b else NA,
                behavioral_baseline_accuracy=ref["baseline_accuracy"] if ref else NA,
                behavioral_accuracy=ref["steered_accuracy"] if ref else NA,
                behavioral_delta=ref["accuracy_delta"] if ref else NA,
                behavioral_fixes=ref["fixes"] if ref else NA,
                behavioral_regressions=ref["regressions"] if ref else NA,
                behavioral_run=b[0]["run_directory"] if b else NA,
                robustness_seed_count=len(observed_seeds),
                robustness_plus7_seed_count=len(multiseed),
                robustness_mean_accuracy=st.mean(x["steered_accuracy"] for x in multiseed)
                if len(multiseed) > 1
                else NA,
                robustness_mean_delta=st.mean(x["accuracy_delta"] for x in multiseed)
                if len(multiseed) > 1
                else NA,
                combined_tested=bool(combined),
                combined_runs=combined,
                notes=(
                    "Behavioral reference is shared coarse-grid +8, not an optimized d"
                    "ose. Robustness means use finalized +7 primary only. Missing mult"
                    "i-seed evidence is not feature rejection."
                ),
            )
        )
    # Compare exact primary artifact hashes and every condition (including combined).
    if primary:
        for check in primary["validation"]["per_run_checks"]:
            item = bypath.get(check["run"])
            if item:
                for name, expected in check["artifact_sha256"].items():
                    if item[0]["artifact_sha256"].get(name) != expected:
                        discrepancies.append(
                            dict(
                                run=check["run"],
                                artifact=name,
                                reason="Hash differs from finalized primary",
                            )
                        )
    # The five-feature +8 table must share the same cases, families and baseline labels.
    reference = None
    for table in tables:
        if not table["behavioral_available"]:
            continue
        objects = read_json(Path(table["behavioral_run"]) / "artifacts/behavioral/generations.json")
        obj = next(x for x in objects if x["coefficient"] == 8.0)
        baseline = {r["case_id"]: (r["family"], r["label"]) for r in obj["baseline_rows"]}
        if reference is None:
            reference = baseline
        elif baseline != reference:
            discrepancies.append(
                dict(
                    feature_id=table["feature_id"],
                    reason="Common +8 comparison baseline/case mismatch",
                )
            )
    duplicates = defaultdict(list)
    for row in inventory:
        result_hashes = {
            k: v
            for k, v in row["artifact_sha256"].items()
            if k.endswith(".csv") or k.endswith("generations.json")
        }
        if result_hashes:
            duplicates[json.dumps(result_hashes, sort_keys=True)].append(row["run_directory"])
    for row in inventory:
        row["same_protocol_attempt_count"] = len(groups[row["run_name"]])
        if not row["selected_for"]:
            row["selection_reason"] = (
                "Inventoried only: prior attempt, different protocol, or non-targe"
                "t control; not substituted for selected evidence."
            )
    configs = []
    for p in sorted((ROOT / "experiments/configs").glob("*.toml")):
        config = tomllib.loads(p.read_text())
        f = config.get("feature", config.get("features", {}))
        ids = f.get("feature_ids", [f["feature_id"]] if "feature_id" in f else [])
        if set(ids) & set(FEATURES) or "discover_topk" in p.stem:
            configs.append(
                dict(
                    path=str(p.resolve()),
                    run_name=config["run"]["name"],
                    feature_ids=ids,
                    status="RUN_PRESENT" if config["run"]["name"] in groups else "CONFIG_ONLY",
                    note=""
                    if config["run"]["name"] in groups
                    else "config exists, run artifact not found",
                )
            )
    return dict(
        analyzer_sha256=digest(Path(__file__)),
        duplicate_result_artifact_groups=[v for v in duplicates.values() if len(v) > 1],
        inventory=inventory,
        feature_table=tables,
        coverage=coverage,
        selection_decisions=decisions,
        discovery=[
            dict(run=x[0]["run_directory"], config=x[2], **x[1]["discover"])
            for x in discovery_runs
            if x
        ],
        causal=causal,
        specificity=specificity,
        behavioral_dose_response=behaviors,
        robustness=robustness,
        all_run_validated_details={r["run_directory"]: d for r, d, c in inspected},
        config_inventory=configs,
        discrepancies=discrepancies,
        counts=dict(
            total_run_directories=len(inventory),
            statuses=dict(Counter(r["status"] for r in inventory)),
            target_related_runs=sum(r["target_related"] for r in inventory),
            selected_unique_runs=sum(bool(r["selected_for"]) for r in inventory),
        ),
        existing_combined_analysis=dict(
            path=str(primary_path),
            sha256=digest(primary_path),
            policy=primary["policy"],
            validation_status=primary["validation"]["status"],
            analyses={
                name: dict(n_observations=a["n_observations"], mean=a["seed_equal_weighted_mean"])
                for name, a in primary["analyses"].items()
            },
        )
        if primary
        else NA,
        limitations=[
            (
                "Completed means artifacts exist and recorded metrics validate, no"
                "t that evidence is positive."
            ),
            (
                "No new labels, models, or experiments. Discovery rank is the init"
                "ial train mean-delta rank, not strong-null selection rank."
            ),
            (
                "Causal/specificity use activation removal coefficient 1; behavior"
                "al uses additive steering, so signs and magnitudes are different "
                "estimands."
            ),
            (
                "Empirical adjusted p=0 is a finite simulation result, not zero po"
                "pulation probability."
            ),
            (
                "Specificity controls reuse the original correct/lure answer strin"
                "gs; positive gap alone does not prove semantic specificity."
            ),
            (
                "Missing detailed historical environments and overlapping seed cas"
                "e sets limit interpretation."
            ),
        ],
    )


def publish(summary, output):
    tables = {
        "topk_feature_evidence_table.csv": summary["feature_table"],
        "topk_feature_run_inventory.csv": summary["inventory"],
        "topk_feature_coverage_matrix.csv": [
            dict(evidence=key, **{str(r["feature_id"]): r[key] for r in summary["coverage"]})
            for key in summary["coverage"][0]
            if key != "feature_id"
        ],
    }
    names = [*tables, "topk_feature_evidence_summary.json"]
    require(
        all(not (output / n).exists() for n in names), "Outputs exist; choose a new --output-dir"
    )
    output.mkdir(parents=True, exist_ok=True)
    published = []
    with tempfile.TemporaryDirectory(dir=output, prefix=".topk-evidence-") as temp:
        for name, rows in tables.items():
            with (Path(temp) / name).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(
                    {
                        k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                        for k, v in r.items()
                    }
                    for r in rows
                )
        (Path(temp) / names[-1]).write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        try:
            for name in names:
                os.link(Path(temp) / name, output / name)
                published.append(output / name)
        except BaseException:
            for p in published:
                p.unlink()
            raise
    return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/analysis")
    args = parser.parse_args()
    summary = analyze(args.runs_dir)
    paths = publish(summary, args.output_dir)
    print(
        json.dumps(
            dict(
                counts=summary["counts"],
                evidence=summary["feature_table"],
                discrepancies=summary["discrepancies"],
                outputs=list(map(str, paths)),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
