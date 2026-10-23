"""CPU stubs for the pinned diagnostic Job and Colab artifact pipeline."""
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from experiments.jobs import matched_branch_margin as job
from experiments.runners import launch_colab as launcher
from experiments.runners.remote_bootstrap import render_bootstrap

CONFIG = job.ROOT / "experiments/configs/diagnose_fresh_v2_matched_branch_margin_f30908_2b.toml"


class MatchedBranchMarginTests(unittest.TestCase):
    def test_script_selection_and_fixed_measurements(self):
        spec = importlib.util.spec_from_file_location("matched_diagnostic", job.SCRIPT)
        diagnostic = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diagnostic)
        self.assertEqual(len(diagnostic.RATE_IDS), 10)
        self.assertEqual(len(diagnostic.GROWTH_IDS), 6)
        selected = diagnostic.RATE_IDS + diagnostic.GROWTH_IDS
        self.assertEqual(len(set(selected)), 16)
        self.assertEqual(diagnostic.COEFFICIENTS, (0.0, 7.0, 8.0))
        cases = {c.case_id: c for c in diagnostic.load_lure_dataset("crt_fresh_v2")}
        self.assertTrue(set(selected) <= cases.keys())
        self.assertTrue(all(cases[cid].family == "crt_rate" for cid in diagnostic.RATE_IDS))
        self.assertTrue(all(cases[cid].family == "crt_growth" for cid in diagnostic.GROWTH_IDS))

    def test_job_calls_only_script_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def fake(command, **kwargs):
                self.assertEqual(command[:3], [job.sys.executable, "-u", str(job.SCRIPT)])
                self.assertEqual(command[5:], ["--layer", "5", "--feature", "30908", "--only", "all"])
                self.assertEqual(kwargs, {"cwd": job.ROOT, "check": True})
                Path(command[4]).write_text("case_id,baseline_margin,plus7_margin,plus8_margin\n")
            with mock.patch.object(job.subprocess, "run", side_effect=fake) as execute:
                run_dir = job.run(CONFIG, root)
                self.assertTrue((run_dir / "diagnostics" / job.CSV_NAME).is_file())
                manifest = json.loads((run_dir / "diagnostics/manifest.json").read_text())
                self.assertEqual(manifest["status"], "succeeded")
                self.assertEqual(manifest["expected_rate_cases"] + manifest["expected_growth_cases"], 16)
                before = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
                with self.assertRaises(FileExistsError):
                    job.run(CONFIG, root)
                execute.assert_called_once()
                self.assertEqual(before, {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()})

    def test_wrong_settings_rejected_without_execution(self):
        text = CONFIG.read_text()
        for old, new in (("layer = 5", "layer = 6"), ("feature = 30908", "feature = 22552"),
                         ("[0.0, 7.0, 8.0]", "[0.0, 7.0]"), ('only = "all"', 'only = "rate"')):
            with self.subTest(new=new), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.toml"
                path.write_text(text.replace(old, new))
                with mock.patch.object(job.subprocess, "run") as execute:
                    with self.assertRaises(ValueError):
                        job.run(path, Path(tmp) / "output")
                    execute.assert_not_called()

    def test_failed_job_records_status(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            job.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["stub"])
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                job.run(CONFIG, Path(tmp))
            manifest = next(Path(tmp).rglob("manifest.json"))
            self.assertEqual(json.loads(manifest.read_text())["status"], "failed")

    def test_dry_run_payload_archive_and_local_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiments/jobs").mkdir(parents=True)
            (root / "scripts").mkdir()
            (root / "experiments/jobs/matched_branch_margin.py").write_text("job source")
            (root / "scripts" / job.SCRIPT.name).write_text("diagnostic source")
            with mock.patch.object(launcher, "timestamp", return_value="fixed"), \
                 mock.patch.object(launcher, "git_value", return_value="stub"), \
                 mock.patch.object(launcher, "ensure_session") as allocate, \
                 mock.patch.object(launcher, "run_cmd") as remote:
                kwargs = dict(root=root, session="test", keep_session=False, default_gpu=None,
                              default_tpu=None, exec_timeout=60, allocation_attempts=1,
                              allocation_retry_delay=0, dry_run=True)
                local = launcher.launch_one(CONFIG, **kwargs)
                manifest = json.loads((local / "launcher_manifest.json").read_text())
                self.assertEqual(manifest["gpu"], "A100")
                self.assertEqual(manifest["payload"]["job_path"],
                                 "experiments/jobs/matched_branch_margin.py")
                self.assertEqual(manifest["payload"]["source"], "archive")
                with zipfile.ZipFile(local / "source.zip") as archive:
                    self.assertIn("scripts/" + job.SCRIPT.name, archive.namelist())
                    self.assertIn("experiments/jobs/matched_branch_margin.py", archive.namelist())
                before = (local / "launcher_manifest.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    launcher.launch_one(CONFIG, **kwargs)
                self.assertEqual(before, (local / "launcher_manifest.json").read_bytes())
                allocate.assert_not_called()
                remote.assert_not_called()

    def test_launcher_uploads_and_downloads_artifacts_without_colab(self):
        for mode in ("success", "fallback", "remote_failure"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                transfers = []

                def transfer(command, **kwargs):
                    transfers.append(command)
                    if command[1] == "download":
                        if mode == "fallback" and command[-2].startswith("/content/mindscopex_artifacts/"):
                            raise subprocess.CalledProcessError(1, command)
                        with zipfile.ZipFile(command[-1], "w") as archive:
                            archive.writestr("diagnostics/" + job.CSV_NAME, "diagnostic CSV")
                            archive.writestr("job.log", "diagnostic log")
                    return subprocess.CompletedProcess(command, 0)

                def remote(command, **kwargs):
                    if command[1] == "log":
                        Path(command[-1]).write_text("Colab log")
                    code = 1 if mode == "remote_failure" and command[1] == "exec" else 0
                    return subprocess.CompletedProcess(command, code)

                with mock.patch.object(launcher, "timestamp", return_value="fixed"), \
                     mock.patch.object(launcher, "git_value", return_value="stub"), \
                     mock.patch.object(launcher, "ensure_session") as allocate, \
                     mock.patch.object(launcher, "run_cmd", side_effect=remote), \
                     mock.patch.object(launcher, "run_cmd_retry", side_effect=transfer):
                    kwargs = dict(root=root, session="test", keep_session=False, default_gpu=None,
                                  default_tpu=None, exec_timeout=60, allocation_attempts=1,
                                  allocation_retry_delay=0, dry_run=False)
                    if mode == "remote_failure":
                        with self.assertRaises(RuntimeError):
                            launcher.launch_one(CONFIG, **kwargs)
                        local = next((root / "results/runs").iterdir())
                    else:
                        local = launcher.launch_one(CONFIG, **kwargs)
                    self.assertEqual(allocate.call_args.args, ("test", "A100", None))
                    uploads = [cmd for cmd in transfers if cmd[1] == "upload"]
                    self.assertEqual(len(uploads), 2)
                    self.assertTrue(uploads[0][-1].endswith("_config.toml"))
                    self.assertTrue(uploads[1][-1].endswith("_source.zip"))
                    downloads = [cmd for cmd in transfers if cmd[1] == "download"]
                    self.assertEqual(len(downloads), 2 if mode == "fallback" else 1)
                    self.assertTrue((local / "artifacts.zip").is_file())
                    self.assertEqual((local / "artifacts/diagnostics" / job.CSV_NAME).read_text(),
                                     "diagnostic CSV")
                    self.assertEqual((local / "artifacts/job.log").read_text(), "diagnostic log")
                    self.assertTrue((local / "colab_log.md").is_file())
                    self.assertTrue((local / "colab_log.jsonl").is_file())

    def bootstrap_namespace(self, root):
        name = job.run_name(job.load_toml(CONFIG))
        payload = dict(remote_repo_dir=str(job.ROOT), remote_output_root=str(root / "output"),
                       remote_config_path=str(CONFIG), run_name=name, job="matched_branch_margin",
                       job_path="experiments/jobs/matched_branch_margin.py")
        source = render_bootstrap(payload).rsplit("\nmain()", 1)[0]
        source = source.replace('Path("/content")', f'Path({str(root)!r})')
        namespace = {}
        exec(compile(source, "<test bootstrap>", "exec"), namespace)
        namespace["prepare_source"] = lambda: job.ROOT
        namespace["ensure_dependencies"] = lambda repo: None
        return namespace, root / "output" / name

    def test_bootstrap_packages_csv_and_logs_on_success_or_failure(self):
        for returncode in (0, 1):
            with self.subTest(returncode=returncode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                namespace, run_dir = self.bootstrap_namespace(root)
                def capture(command, log_path):
                    self.assertIn("experiments/jobs/matched_branch_margin.py", str(command[1]))
                    self.assertNotIn("research_experiments.py", str(command))
                    log_path.write_text("diagnostic stdout and stderr\n")
                    diagnostics = run_dir / "diagnostics"
                    diagnostics.mkdir()
                    (diagnostics / job.CSV_NAME).write_text("partial diagnostic CSV\n")
                    return returncode
                namespace["run_capturing"] = capture
                with mock.patch.dict(os.environ), mock.patch("os.chdir"):
                    if returncode:
                        with self.assertRaises(SystemExit):
                            namespace["main"]()
                    else:
                        namespace["main"]()
                with zipfile.ZipFile(run_dir.with_suffix(".zip")) as archive:
                    self.assertIn("job.log", archive.namelist())
                    self.assertIn("diagnostics/" + job.CSV_NAME, archive.namelist())
                    self.assertIn("stdout and stderr", archive.read("job.log").decode())
                self.assertTrue((root / f"{run_dir.name}_artifacts.zip").is_file())

    def test_bootstrap_rejects_existing_artifacts_before_opening_log(self):
        for kind in ("run_dir", "zip", "fallback", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                namespace, run_dir = self.bootstrap_namespace(root)
                run_dir.parent.mkdir()
                if kind == "run_dir":
                    run_dir.mkdir()
                    target = run_dir / "job.log"
                elif kind == "zip":
                    target = run_dir.with_suffix(".zip")
                else:
                    target = root / f"{run_dir.name}_artifacts.zip"
                target.write_bytes(b"preserve")
                if kind == "symlink":
                    run_dir.symlink_to(root / "missing", target_is_directory=True)
                capture = mock.Mock()
                namespace["run_capturing"] = capture
                with mock.patch.dict(os.environ), mock.patch("os.chdir"), \
                     self.assertRaises(FileExistsError):
                    namespace["main"]()
                capture.assert_not_called()
                self.assertEqual(target.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
