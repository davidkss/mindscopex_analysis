"""Regression coverage for the discovery localization artifact schema (CPU only)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.jobs import research_experiments as job


class LocalizationPlotTests(unittest.TestCase):
    def test_screen_null_schema_renders_png_and_preserves_values(self) -> None:
        # Localization uses screen_* fields; null_mean belongs to null_summary.
        row = {
            "layer": 11,
            "feature_id": 19069,
            "mean_margin_delta": 0.2,  # synthetic discovery value
            "screen_hostile_delta": 0.1,  # synthetic screen value
            "screen_null_mean": 0.042401153982306525,
            "screen_null_z": 0.33710183221051554,
            "screen_null_percentile": 0.75,
        }
        original = dict(row)
        fig, ax = job.plt.subplots()
        self.addCleanup(job.plt.close, fig)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "discover_localization.png"
            with mock.patch.object(job.plt, "subplots", return_value=(fig, ax)):
                job._plot_localization([row], path)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(list(ax.lines[0].get_xdata()), [11])
        self.assertEqual(list(ax.lines[0].get_ydata()), [row["mean_margin_delta"]])
        self.assertEqual(list(ax.lines[1].get_ydata()), [row["screen_null_mean"]])
        self.assertEqual(row, original)
