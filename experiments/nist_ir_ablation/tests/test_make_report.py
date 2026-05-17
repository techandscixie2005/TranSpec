"""
Tests for make_report.py using temporary summary files.

Creates fake summary_mean_std.csv and ablation_effects.json, then runs
make_report.py and verifies output.
"""

import csv
import json
import os
import sys
import tempfile

import pytest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_SCRIPT_DIR, "..", "scripts")
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
for p in [_SCRIPTS_DIR, _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


def create_fake_summary(summary_dir):
    """Create fake summary CSV and effects JSON."""
    # summary_mean_std.csv
    metrics = ["top1", "top3", "top5", "top10", "invalid_decode_rate",
               "avg_num_candidates", "total_params", "train_time_sec", "eval_time_sec"]
    fieldnames = ["model_id"] + [f"{m}_{stat}" for m in metrics for stat in ("mean", "std")]

    rows = [
        {
            "model_id": "E0_atom_nope",
            "top1_mean": "0.3500", "top1_std": "0.0100",
            "top3_mean": "0.5200", "top3_std": "0.0150",
            "top5_mean": "0.6100", "top5_std": "0.0110",
            "top10_mean": "0.7300", "top10_std": "0.0090",
            "invalid_decode_rate_mean": "0.0500", "invalid_decode_rate_std": "0.0100",
            "avg_num_candidates_mean": "50.0000", "avg_num_candidates_std": "5.0000",
            "total_params_mean": "12345678", "total_params_std": "0",
            "train_time_sec_mean": "100.0000", "train_time_sec_std": "10.0000",
            "eval_time_sec_mean": "50.0000", "eval_time_sec_std": "5.0000",
        },
        {
            "model_id": "E1_atom_fourier",
            "top1_mean": "0.3700", "top1_std": "0.0120",
            "top3_mean": "0.5400", "top3_std": "0.0150",
            "top5_mean": "0.6300", "top5_std": "0.0110",
            "top10_mean": "0.7500", "top10_std": "0.0090",
            "invalid_decode_rate_mean": "0.0400", "invalid_decode_rate_std": "0.0100",
            "avg_num_candidates_mean": "52.0000", "avg_num_candidates_std": "4.0000",
            "total_params_mean": "12456789", "total_params_std": "0",
            "train_time_sec_mean": "105.0000", "train_time_sec_std": "8.0000",
            "eval_time_sec_mean": "52.0000", "eval_time_sec_std": "4.0000",
        },
        {
            "model_id": "E2_spe_nope",
            "top1_mean": "0.3300", "top1_std": "0.0110",
            "top3_mean": "0.5000", "top3_std": "0.0140",
            "top5_mean": "0.5900", "top5_std": "0.0100",
            "top10_mean": "0.7100", "top10_std": "0.0080",
            "invalid_decode_rate_mean": "0.0600", "invalid_decode_rate_std": "0.0120",
            "avg_num_candidates_mean": "48.0000", "avg_num_candidates_std": "6.0000",
            "total_params_mean": "12567890", "total_params_std": "0",
            "train_time_sec_mean": "110.0000", "train_time_sec_std": "12.0000",
            "eval_time_sec_mean": "55.0000", "eval_time_sec_std": "6.0000",
        },
        {
            "model_id": "E3_spe_fourier",
            "top1_mean": "0.3600", "top1_std": "0.0130",
            "top3_mean": "0.5300", "top3_std": "0.0160",
            "top5_mean": "0.6200", "top5_std": "0.0120",
            "top10_mean": "0.7400", "top10_std": "0.0100",
            "invalid_decode_rate_mean": "0.0450", "invalid_decode_rate_std": "0.0110",
            "avg_num_candidates_mean": "51.0000", "avg_num_candidates_std": "5.0000",
            "total_params_mean": "12678901", "total_params_std": "0",
            "train_time_sec_mean": "108.0000", "train_time_sec_std": "9.0000",
            "eval_time_sec_mean": "53.0000", "eval_time_sec_std": "5.0000",
        },
    ]

    csv_path = os.path.join(summary_dir, "summary_mean_std.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ablation_effects.json
    effects = {
        "top1": {
            "delta_pe": 0.02,
            "delta_spe": -0.02,
            "delta_both": 0.01,
            "interaction": 0.01,
        },
        "top3": {
            "delta_pe": 0.02,
            "delta_spe": -0.02,
            "delta_both": 0.01,
            "interaction": 0.01,
        },
        "top5": {
            "delta_pe": 0.02,
            "delta_spe": -0.02,
            "delta_both": 0.01,
            "interaction": 0.01,
        },
        "top10": {
            "delta_pe": 0.02,
            "delta_spe": -0.02,
            "delta_both": 0.01,
            "interaction": 0.01,
        },
    }
    effects_path = os.path.join(summary_dir, "ablation_effects.json")
    with open(effects_path, "w") as f:
        json.dump(effects, f, indent=2)

    # missing_runs.json (no missing)
    missing_path = os.path.join(summary_dir, "missing_runs.json")
    with open(missing_path, "w") as f:
        json.dump({"missing": []}, f, indent=2)

    return csv_path, effects_path, missing_path


class TestMakeReport:
    """Test make_report.py functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.summary_dir = os.path.join(self.tmpdir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
        create_fake_summary(self.summary_dir)
        self.output_md = os.path.join(self.tmpdir, "report.md")
        self.output_html = os.path.join(self.tmpdir, "report.html")
        yield
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_report_md_created(self):
        """Test that make_report.py creates report.md."""
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "make_report.py"),
                "--summary_dir", self.summary_dir,
                "--output_md", self.output_md,
                "--output_html", self.output_html,
            ],
            capture_output=True, text=True,
        )
        print(f"stdout: {result.stdout}")
        if result.returncode != 0:
            print(f"stderr: {result.stderr}")
        assert result.returncode == 0
        assert os.path.exists(self.output_md)

    def test_report_html_created(self):
        """Test that make_report.py creates report.html."""
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "make_report.py"),
                "--summary_dir", self.summary_dir,
                "--output_md", self.output_md,
                "--output_html", self.output_html,
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert os.path.exists(self.output_html)

    def test_report_content_structure(self):
        """Test that report.md contains expected sections."""
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "make_report.py"),
                "--summary_dir", self.summary_dir,
                "--output_md", self.output_md,
                "--output_html", self.output_html,
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

        with open(self.output_md) as f:
            content = f.read()

        # Check main sections exist
        assert "# NIST IR Ablation Experiment Report" in content
        assert "## Experiment Matrix" in content
        assert "## Main Results" in content
        assert "## Ablation Effects" in content
        assert "## Reproduction Commands" in content
        assert "## Preliminary Conclusions" in content

        # Check model names appear
        assert "E0_atom_nope" in content
        assert "E1_atom_fourier" in content
        assert "E2_spe_nope" in content
        assert "E3_spe_fourier" in content

        # Check effects
        assert "Fourier PE (E1 − E0)" in content
        assert "SPE (E2 − E0)" in content
        assert "Both (E3 − E0)" in content
        assert "Interaction" in content

        # Check values
        assert "0.3500" in content
        assert "0.3700" in content

    def test_report_html_valid(self):
        """Test that report.html is valid HTML."""
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "make_report.py"),
                "--summary_dir", self.summary_dir,
                "--output_md", self.output_md,
                "--output_html", self.output_html,
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

        with open(self.output_html) as f:
            content = f.read()

        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "<h1>" in content
        assert "</body>" in content
        assert "</html>" in content

    def test_report_with_missing_runs(self):
        """Test report generation with missing runs."""
        # Create missing_runs.json with entries
        missing_path = os.path.join(self.summary_dir, "missing_runs.json")
        with open(missing_path, "w") as f:
            json.dump({
                "missing": [
                    {"model": "E3_spe_fourier", "seed": "3407"},
                ]
            }, f, indent=2)

        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "make_report.py"),
                "--summary_dir", self.summary_dir,
                "--output_md", self.output_md,
                "--output_html", self.output_html,
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

        with open(self.output_md) as f:
            content = f.read()
        assert "Missing Runs" in content
        assert "E3_spe_fourier" in content

    def test_report_no_effects_file(self):
        """Test report generation when effects file is missing."""
        os.remove(os.path.join(self.summary_dir, "ablation_effects.json"))

        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "make_report.py"),
                "--summary_dir", self.summary_dir,
                "--output_md", self.output_md,
                "--output_html", self.output_html,
            ],
            capture_output=True, text=True,
        )
        # Should still succeed without crash
        assert result.returncode == 0
        with open(self.output_md) as f:
            content = f.read()
        assert "No effects data" in content or "No ablation effects" in content
