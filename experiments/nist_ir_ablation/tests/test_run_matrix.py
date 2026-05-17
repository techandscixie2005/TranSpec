"""
Tests for run_matrix.py command construction and dry-run mode.

These tests use --dry_run to verify command construction without
actually executing training/evaluation.
"""

import os
import subprocess
import sys

import pytest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_SCRIPT_DIR, "..", "scripts")
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))


class TestRunMatrixDryRun:
    """Test run_matrix.py in dry-run mode."""

    def test_dry_run_basic(self):
        """Test that dry-run mode works with basic arguments."""
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "run_matrix.py"),
                "--dry_run",
                "--input", "/fake/input.jsonl",
                "--output", "/tmp/fake_output",
                "--config", os.path.join(_PROJECT_ROOT,
                    "experiments/nist_ir_ablation/configs/smoke_200.yaml"),
                "--models", "E0_atom_nope", "E1_atom_fourier",
                "--seeds", "42",
                "--run_preprocess",
                "--run_train",
                "--run_eval",
                "--aggregate",
            ],
            capture_output=True, text=True,
        )
        print(f"stdout: {result.stdout[:1000]}")
        if result.returncode != 0:
            print(f"stderr: {result.stderr}")
        assert result.returncode == 0

        # Verify output contains expected dry-run markers
        assert "[DRY-RUN]" in result.stdout
        assert "NIST IR Ablation Experiment Matrix" in result.stdout
        assert "E0_atom_nope" in result.stdout
        assert "E1_atom_fourier" in result.stdout

    def test_dry_run_single_model(self):
        """Test dry-run with single --model specification."""
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "run_matrix.py"),
                "--dry_run",
                "--output", "/tmp/fake_output",
                "--config", os.path.join(_PROJECT_ROOT,
                    "experiments/nist_ir_ablation/configs/smoke_200.yaml"),
                "--model", "E0_atom_nope",
                "--seed", "42",
                "--run_train",
                "--run_eval",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "[DRY-RUN]" in result.stdout
        assert "E0_atom_nope" in result.stdout

    def test_dry_run_all_four_conditions(self):
        """Test dry-run with all four model conditions."""
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "run_matrix.py"),
                "--dry_run",
                "--output", "/tmp/fake_output",
                "--config", os.path.join(_PROJECT_ROOT,
                    "experiments/nist_ir_ablation/configs/smoke_200.yaml"),
                "--models",
                "E0_atom_nope", "E1_atom_fourier",
                "E2_spe_nope", "E3_spe_fourier",
                "--seeds", "42",
                "--run_train",
                "--run_eval",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        # Should mention all four conditions
        for model in ["E0_atom_nope", "E1_atom_fourier", "E2_spe_nope", "E3_spe_fourier"]:
            assert model in result.stdout

    def test_dry_run_no_models(self):
        """Test that missing models/seed cause error."""
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "run_matrix.py"),
                "--dry_run",
                "--output", "/tmp/fake_output",
                "--config", os.path.join(_PROJECT_ROOT,
                    "experiments/nist_ir_ablation/configs/smoke_200.yaml"),
            ],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_dry_run_continue_on_error(self):
        """Test continue_on_error flag doesn't crash."""
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "run_matrix.py"),
                "--dry_run",
                "--output", "/tmp/fake_output",
                "--config", os.path.join(_PROJECT_ROOT,
                    "experiments/nist_ir_ablation/configs/smoke_200.yaml"),
                "--model", "E0_atom_nope",
                "--seed", "42",
                "--run_train",
                "--run_eval",
                "--continue_on_error",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_dry_run_preprocess_only(self):
        """Test preprocessing-only mode."""
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "run_matrix.py"),
                "--dry_run",
                "--input", "/fake/input.jsonl",
                "--output", "/tmp/fake_output",
                "--config", os.path.join(_PROJECT_ROOT,
                    "experiments/nist_ir_ablation/configs/smoke_200.yaml"),
                "--models", "E0_atom_nope",
                "--seeds", "42",
                "--run_preprocess",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "[DRY-RUN]" in result.stdout
        assert "prepare_jsonl.py" in result.stdout

    def test_dry_run_skip_preprocess_if_exists(self):
        """Test skip_preprocess_if_exists flag."""
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_SCRIPTS_DIR, "run_matrix.py"),
                "--dry_run",
                "--input", "/fake/input.jsonl",
                "--output", "/tmp/fake_output",
                "--config", os.path.join(_PROJECT_ROOT,
                    "experiments/nist_ir_ablation/configs/smoke_200.yaml"),
                "--models", "E0_atom_nope",
                "--seeds", "42",
                "--run_preprocess",
                "--skip_preprocess_if_exists",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_condition_config_paths(self):
        """Test that condition config paths resolve correctly."""
        # Check that each model's YAML exists
        for model in ["E0_atom_nope", "E1_atom_fourier", "E2_spe_nope", "E3_spe_fourier"]:
            config_path = os.path.join(
                _PROJECT_ROOT, "experiments/nist_ir_ablation/configs", f"{model}.yaml"
            )
            assert os.path.exists(config_path), f"Missing config: {config_path}"

    def test_script_imports(self):
        """Test that run_matrix.py can be imported without errors."""
        # Run a syntax check
        result = subprocess.run(
            [sys.executable, "-c",
             f"import py_compile; py_compile.compile('{os.path.join(_SCRIPTS_DIR, 'run_matrix.py')}', doraise=True)"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"
