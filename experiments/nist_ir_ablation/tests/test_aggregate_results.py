"""
Tests for aggregate_results.py using a temporary fake run directory.

Because aggregate_results.py operates on files (reading metrics.json,
writing CSVs), we create a temporary directory with fake run outputs.
"""

import json
import os
import sys
import tempfile
import csv

import pytest

# Add project root and scripts directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_SCRIPT_DIR, "..", "scripts")
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
for p in [_SCRIPTS_DIR, _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Re-import here so we can also use the script
import importlib


def make_fake_metrics(model_id, seed, top1=0.35, top3=0.52, top5=0.61, top10=0.73,
                      invalid_decode_rate=0.05, avg_num_candidates=50.0,
                      total_params=12345678, trainable_params=10000000,
                      train_time_sec=100.0, eval_time_sec=50.0):
    """Create a fake metrics.json dict."""
    return {
        "model_id": model_id,
        "seed": seed,
        "num_test": 562,
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "top10": top10,
        "invalid_decode_count": int(avg_num_candidates * invalid_decode_rate * 562),
        "invalid_decode_rate": invalid_decode_rate,
        "avg_num_candidates": avg_num_candidates,
        "decode_method": "beam",
        "beam_size": 10,
        "candidate_limit": 10,
        "max_decode_len": 256,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_time_sec": train_time_sec,
        "eval_time_sec": eval_time_sec,
        "checkpoint_path": f"runs/{model_id}/seed_{seed}/checkpoints/best.pt",
        "processed_dir": "processed",
    }


def create_fake_run_tree(base_dir, runs_data):
    """Create a directory tree with fake metrics.json files.

    runs_data: dict of (model_id, seed) -> metrics dict
    """
    for (model_id, seed), metrics in runs_data.items():
        run_dir = os.path.join(base_dir, model_id, f"seed_{seed}")
        eval_dir = os.path.join(run_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)


class TestAggregateResults:
    """Test aggregate_results.py functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create a temporary directory with fake run data."""
        self.tmpdir = tempfile.mkdtemp()
        self.run_dir = os.path.join(self.tmpdir, "runs")
        self.summary_dir = os.path.join(self.tmpdir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)

        # Create 4 fake runs
        self.runs_data = {
            ("E0_atom_nope", 42): make_fake_metrics("E0_atom_nope", 42, top1=0.35, top3=0.52, top5=0.61, top10=0.73),
            ("E1_atom_fourier", 42): make_fake_metrics("E1_atom_fourier", 42, top1=0.37, top3=0.54, top5=0.63, top10=0.75),
            ("E2_spe_nope", 42): make_fake_metrics("E2_spe_nope", 42, top1=0.33, top3=0.50, top5=0.59, top10=0.71),
            ("E3_spe_fourier", 42): make_fake_metrics("E3_spe_fourier", 42, top1=0.36, top3=0.53, top5=0.62, top10=0.74),
        }
        create_fake_run_tree(self.run_dir, self.runs_data)

        yield

        # Cleanup
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_discover_runs(self):
        """Test that discover_runs finds all created runs."""
        from aggregate_results import discover_runs
        runs, missing = discover_runs(self.run_dir)
        assert len(runs) == 4
        assert len(missing) == 8  # 12 expected - 4 found = 8 missing (other seeds)
        assert ("E0_atom_nope", 42) in runs
        assert runs[("E0_atom_nope", 42)]["top1"] == 0.35

    def test_aggregate_by_model(self):
        """Test that aggregate_by_model computes correct stats."""
        from aggregate_results import aggregate_by_model
        runs, _ = discover_runs(self.run_dir)
        summary = aggregate_by_model(runs)

        assert "E0_atom_nope" in summary
        assert summary["E0_atom_nope"]["runs"] == 1
        assert summary["E0_atom_nope"]["top1_mean"] == 0.35

    def test_compute_mean_std(self):
        """Test compute_mean_std with single and multiple values."""
        from aggregate_results import compute_mean_std
        mean, std = compute_mean_std([1.0])
        assert mean == 1.0
        assert std == 0.0

        mean, std = compute_mean_std([1.0, 2.0, 3.0])
        assert mean == 2.0
        assert std == 1.0

    def test_compute_ablation_effects(self):
        """Test that ablation effects are computed correctly."""
        from aggregate_results import aggregate_by_model, compute_ablation_effects
        runs, _ = discover_runs(self.run_dir)
        summary = aggregate_by_model(runs)
        effects = compute_ablation_effects(summary)

        # delta_pe = E1 - E0 = 0.37 - 0.35 = 0.02
        assert effects["top1"]["delta_pe"] == 0.02

        # delta_spe = E2 - E0 = 0.33 - 0.35 = -0.02
        assert effects["top1"]["delta_spe"] == -0.02

        # delta_both = E3 - E0 = 0.36 - 0.35 = 0.01
        assert effects["top1"]["delta_both"] == 0.01

        # interaction = (E3 - E2) - (E1 - E0) = (0.36-0.33) - (0.37-0.35) = 0.03 - 0.02 = 0.01
        assert effects["top1"]["interaction"] == 0.01

    def test_write_all_runs_csv(self):
        """Test that all_runs.csv is written correctly."""
        from aggregate_results import discover_runs, write_all_runs_csv
        runs, _ = discover_runs(self.run_dir)
        path = os.path.join(self.summary_dir, "all_runs.csv")
        write_all_runs_csv(runs, path)

        assert os.path.exists(path)
        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Header + 4 data rows
        assert len(rows) == 5
        assert rows[0][0] == "model_id"

    def test_write_summary_csv(self):
        """Test that summary_mean_std.csv is written correctly."""
        from aggregate_results import aggregate_by_model, write_summary_csv
        runs, _ = discover_runs(self.run_dir)
        summary = aggregate_by_model(runs)
        path = os.path.join(self.summary_dir, "summary_mean_std.csv")
        write_summary_csv(summary, path)

        assert os.path.exists(path)
        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Header + 4 data rows
        assert len(rows) == 5

        # Check headers
        assert "top1_mean" in rows[0]
        assert "top1_std" in rows[0]

    def test_ablation_effects_json(self):
        """Test that ablation_effects.json is written correctly via main()."""
        # Run the aggregate script as a subprocess
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "aggregate_results.py"),
                "--run_dir", self.run_dir,
                "--summary_dir", self.summary_dir,
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

        # Check output files
        assert os.path.exists(os.path.join(self.summary_dir, "all_runs.csv"))
        assert os.path.exists(os.path.join(self.summary_dir, "summary_mean_std.csv"))
        assert os.path.exists(os.path.join(self.summary_dir, "ablation_effects.json"))
        assert os.path.exists(os.path.join(self.summary_dir, "missing_runs.json"))

        # Verify ablation effects content
        with open(os.path.join(self.summary_dir, "ablation_effects.json")) as f:
            effects = json.load(f)
        assert "top1" in effects
        assert "delta_pe" in effects["top1"]
        assert "delta_spe" in effects["top1"]

    def test_missing_runs_detection(self):
        """Test that missing runs are reported."""
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "aggregate_results.py"),
                "--run_dir", self.run_dir,
                "--summary_dir", self.summary_dir,
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

        with open(os.path.join(self.summary_dir, "missing_runs.json")) as f:
            missing_data = json.load(f)
        assert len(missing_data["missing"]) > 0


class TestAggregateResultsFullMatrix:
    """Test with a full 12-run matrix."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_dir = os.path.join(self.tmpdir, "runs")
        self.summary_dir = os.path.join(self.tmpdir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)

        models = ["E0_atom_nope", "E1_atom_fourier", "E2_spe_nope", "E3_spe_fourier"]
        seeds = [42, 2025, 3407]
        runs_data = {}
        for model in models:
            for i, seed in enumerate(seeds):
                base_top1 = {"E0_atom_nope": 0.35, "E1_atom_fourier": 0.37,
                             "E2_spe_nope": 0.33, "E3_spe_fourier": 0.36}[model]
                noise = i * 0.01
                runs_data[(model, seed)] = make_fake_metrics(
                    model, seed, top1=base_top1 + noise,
                    top3=base_top1 + 0.17 + noise,
                    top5=base_top1 + 0.26 + noise,
                    top10=base_top1 + 0.38 + noise,
                )
        create_fake_run_tree(self.run_dir, runs_data)

        yield

        import shutil
        shutil.rmtree(self.tmpdir)

    def test_full_matrix_aggregation(self):
        """Test aggregation with full 12-run matrix."""
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "aggregate_results.py"),
                "--run_dir", self.run_dir,
                "--summary_dir", self.summary_dir,
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

        # Should have all runs
        with open(os.path.join(self.summary_dir, "all_runs.csv")) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 13  # header + 12 runs

        # No missing runs
        with open(os.path.join(self.summary_dir, "missing_runs.json")) as f:
            missing_data = json.load(f)
        assert len(missing_data["missing"]) == 0

        # Strict mode should pass
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "aggregate_results.py"),
                "--run_dir", self.run_dir,
                "--summary_dir", self.summary_dir,
                "--strict",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_strict_mode_fails_with_missing(self):
        """Test that --strict fails when runs are missing."""
        import subprocess
        # Remove one run
        import shutil
        shutil.rmtree(os.path.join(self.run_dir, "E3_spe_fourier", "seed_3407"), ignore_errors=True)

        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "aggregate_results.py"),
                "--run_dir", self.run_dir,
                "--summary_dir", self.summary_dir,
                "--strict",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_empty_run_dir(self):
        """Test behavior with empty run directory."""
        import subprocess
        empty_dir = tempfile.mkdtemp()
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "aggregate_results.py"),
                "--run_dir", empty_dir,
                "--summary_dir", os.path.join(empty_dir, "summary"),
            ],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        import shutil
        shutil.rmtree(empty_dir)


# Helper for direct usage
def discover_runs(run_dir):
    """Inline discover_runs for testing without importing."""
    from aggregate_results import discover_runs
    return discover_runs(run_dir)
