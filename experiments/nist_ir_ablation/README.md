# NIST IR Ablation Experiment

## Purpose

A 2x2 ablation experiment evaluating two design choices for SMILES prediction from experimental IR spectra:

| Dimension | Variant A | Variant B |
|---|---|---|
| **Spectral encoding** | Intensity-only (1 channel) | Intensity + Fourier features (65 channels) |
| **SMILES tokenization** | Atom-level (character split) | SPE (BPE subword) |

The four conditions are:

| Condition | Spectral Encoding | Tokenizer | Config |
|---|---|---|---|
| E0 (atom_nope) | Intensity only | Atom-level | `configs/E0_atom_nope.yaml` |
| E1 (atom_fourier) | Intensity + Fourier | Atom-level | `configs/E1_atom_fourier.yaml` |
| E2 (spe_nope) | Intensity only | SPE (BPE subword) | `configs/E2_spe_nope.yaml` |
| E3 (spe_fourier) | Intensity + Fourier | SPE (BPE subword) | `configs/E3_spe_fourier.yaml` |

Each condition is trained across 3 random seeds (42, 2025, 3407) for a total of 12 runs, then results are aggregated to compute mean/std Top-1/3/5/10 accuracy and ablation effects (delta PE, delta SPE, interaction).

---

## Input JSONL Format

Each line is a JSON object with the following fields:

```json
{
  "smiles": "CCO",
  "value": {
    "x": [629.9, 650.2, ...],
    "y": [0.012, 0.015, ...]
  }
}
```

- **`smiles`** (string, required): SMILES string of the molecule.
- **`value.x`** (array of float, required): Wavenumber values in cm^-1.
- **`value.y`** (array of float, required): Intensity values (arbitrary units).

Notes:
- Wavenumber ranges vary per record, typically ~630--4835 cm^-1.
- Number of points varies per record, typically 2022--4872.
- Spectra are resampled to a fixed 3000-point grid over [552, 3844] cm^-1 via cubic interpolation during preprocessing.
- Optional fields (`temperature`, `pressure`, `condition`, `type`, `source`) are preserved as metadata.

---

## Directory Layout

```
experiments/nist_ir_ablation/
  configs/              # YAML configuration files
    nist_ir_base.yaml   # Base config (model architecture, training, decoding)
    smoke_200.yaml      # Smoke test overrides (smaller model, fewer epochs)
    E0_atom_nope.yaml   # Condition-specific overrides
    E1_atom_fourier.yaml
    E2_spe_nope.yaml
    E3_spe_fourier.yaml
  transpec_ext/         # Core modules (dataset, tokenizers, encoding, metrics)
  scripts/              # Pipeline scripts
    prepare_jsonl.py    # Preprocessing: read JSONL, canonicalize SMILES, build vocab, split
    train_one.py        # Single model training
    evaluate_one.py     # Single model evaluation
    run_matrix.py       # Orchestrator for full experiment matrix
    aggregate_results.py # Aggregate results across runs
    make_report.py      # Generate report.md and report.html
  tests/                # Unit tests
  eval_output/          # (output) Per-model evaluation artifacts
```

At runtime, the output root (e.g., `runs/nist_ir_ablation/`) contains:

```
runs/nist_ir_ablation/
  processed/
    common/         # Shared: records.pt, split_indices.json, preprocess_manifest.json
    atom/           # Atom-tokenizer encoded train/valid/test .pt files
    spe/            # SPE-tokenizer encoded train/valid/test .pt files
  runs/
    E0_atom_nope/seed_42/     # Per-model, per-seed run directories
      checkpoints/best.pt     # Best checkpoint (by validation loss)
      eval/metrics.json       # Evaluation metrics
      logs/train_stdout.log
      logs/eval_stdout.log
    E0_atom_nope/seed_2025/
    ...
  summary/
    report.md                 # Final experiment report
    report.html               # HTML version
    all_runs.csv              # All 12 runs as rows
    summary_mean_std.csv      # Per-model mean +/- std
    ablation_effects.json     # Delta PE, delta SPE, interaction effects
    missing_runs.json         # Any incomplete runs
```

---

## Smoke Test

Quick validation with 200 samples, 1 seed, reduced model size, and 2 epochs:

```bash
bash experiments/nist_ir_ablation/run_smoke_200.sh \
  --input data/nist_ir/raw/IR_nist_200.jsonl \
  --output runs/nist_ir_ablation_smoke
```

The smoke test runs all 4 conditions with seed=42, using the `smoke_200.yaml` config (d_model=64, 2 layers, 2 epochs, no AMP).

---

## Full Sequential Run

To run the complete 12-run experiment (4 models x 3 seeds) sequentially:

```bash
bash experiments/nist_ir_ablation/run_full_12.sh \
  --input data/nist_ir/raw/IR_nist.jsonl \
  --output runs/nist_ir_ablation
```

This runs preprocessing, all 12 train+eval jobs, then aggregation and report generation. Expect several hours to complete on a GPU.

---

## Slurm Workflow

For cluster execution, use a three-step workflow:

### Step 1: Preprocess

```bash
python experiments/nist_ir_ablation/scripts/prepare_jsonl.py \
  --input data/nist_ir/raw/IR_nist.jsonl \
  --output runs/nist_ir_ablation \
  --config experiments/nist_ir_ablation/configs/nist_ir_base.yaml
```

This reads the JSONL, canonicalizes SMILES, resamples spectra, splits into train/valid/test, and builds both atom and SPE tokenizers. Output is written to `runs/nist_ir_ablation/processed/`.

### Step 2: Submit Slurm array

```bash
sbatch experiments/nist_ir_ablation/run_full_12.slurm
```

The Slurm array job launches 12 parallel tasks (4 models x 3 seeds), each training and evaluating independently. Each task uses its own GPU.

### Step 3: Aggregate and generate report

After all array tasks complete:

```bash
python experiments/nist_ir_ablation/scripts/aggregate_results.py \
  --run_dir runs/nist_ir_ablation/runs \
  --summary_dir runs/nist_ir_ablation/summary

python experiments/nist_ir_ablation/scripts/make_report.py \
  --summary_dir runs/nist_ir_ablation/summary \
  --output_md runs/nist_ir_ablation/summary/report.md \
  --output_html runs/nist_ir_ablation/summary/report.html
```

---

## N16R4 GPU Partition (Beijing Super Cloud Computing Center Lingshui)

Specific workflow for the N16R4 GPU partition:

### Step 1: Checkout branch and prepare data

```bash
git checkout nist-ir-ablation-phase12
```

Place the input file at `data/nist_ir/raw/IR_nist.jsonl`. See [Input Format](#input-format) above for the expected JSONL schema.

### Step 2: Preprocess data (run once, from login node)

```bash
python experiments/nist_ir_ablation/scripts/prepare_jsonl.py \
  --input data/nist_ir/raw/IR_nist.jsonl \
  --output runs/nist_ir_ablation \
  --config experiments/nist_ir_ablation/configs/nist_ir_base.yaml
```

This writes processed data to `runs/nist_ir_ablation/processed/`.

### Step 3: Submit Slurm array

```bash
sbatch experiments/nist_ir_ablation/run_full_12.slurm
```

The Slurm script (`run_full_12.slurm`) is pre-configured for N16R4:
- `#SBATCH -p gpu`
- `#SBATCH --gpus=1` (one GPU per array task)
- `#SBATCH --array=0-11` (12 parallel tasks, one per model/seed)
- Output logs: `slurm_logs/nist_ir_ablation/%A_%a.{out,err}`

Before submitting, edit the environment setup in the script if needed:
```bash
# Uncomment and adjust for your cluster:
# module load miniforge/24.11
# source activate transpec
```

### Step 4: Aggregate and generate report

After all array jobs complete:

```bash
python experiments/nist_ir_ablation/scripts/aggregate_results.py \
  --run_dir runs/nist_ir_ablation/runs \
  --summary_dir runs/nist_ir_ablation/summary

python experiments/nist_ir_ablation/scripts/make_report.py \
  --summary_dir runs/nist_ir_ablation/summary \
  --output_md runs/nist_ir_ablation/summary/report.md \
  --output_html runs/nist_ir_ablation/summary/report.html
```

---

## Run Aggregation Only

If the 12 runs are already completed (checkpoints and eval/metrics.json exist), you can skip training and evaluation and run only aggregation:

```bash
python experiments/nist_ir_ablation/scripts/aggregate_results.py \
  --run_dir runs/nist_ir_ablation/runs \
  --summary_dir runs/nist_ir_ablation/summary

python experiments/nist_ir_ablation/scripts/make_report.py \
  --summary_dir runs/nist_ir_ablation/summary \
  --output_md runs/nist_ir_ablation/summary/report.md \
  --output_html runs/nist_ir_ablation/summary/report.html
```

Add `--strict` to `aggregate_results.py` to require all 12 runs (exits with error if any are missing).

---

## Report Output

The final report is written to `runs/nist_ir_ablation/summary/report.md`. It includes:

- **Dataset info**: Input file, total records, train/valid/test split sizes.
- **Experiment matrix**: The 2x2 ablation conditions and their descriptions.
- **Main results table**: Per-model Top-1/3/5/10 accuracy with mean +/- std across seeds.
- **Ablation effects**: Delta PE, delta SPE, delta Both, and interaction effect for each Top-k metric.
- **Missing runs**: List of any incomplete or missing runs.
- **Reproduction commands**: Preprocessing, sequential run, Slurm workflow, and smoke test commands.
- **Preliminary conclusions**: Directional observations about Fourier PE, SPE, and their interaction.

An HTML version (`report.html`) is also generated for easier viewing.

---

## Git-Ignored Files

The following are intentionally excluded from version control (see `.gitignore`):

- `runs/` -- All experiment output directories
- `data/` -- Large data files
- `*.pt`, `*.pth`, `*.ckpt` -- Model checkpoints and PyTorch tensors
- `*.log`, `*.out`, `*.err` -- Log files
- `__pycache__/`, `*.pyc` -- Python bytecode
- `slurm_logs/` -- Slurm output logs

Checkpoints are large binary files (~100 MB each). Never commit them.

---

## Important Warning

Full runs (12+ experiments on the full NIST dataset) should be executed on a server with a GPU. Running on a local workstation is not recommended -- expect 1-4 hours per training run depending on GPU, totaling 12-48 hours for the full matrix.

Use the smoke test (200 samples, 1 seed, 2 epochs, 8 batch size) for local validation. It completes in a few minutes on any CUDA-capable GPU.

---

## Troubleshooting

### Missing RDKit

```
ModuleNotFoundError: No module named 'rdkit'
```

Install RDKit:

```bash
conda install -c conda-forge rdkit
# or
pip install rdkit-pypi
```

### Missing scipy

```
ModuleNotFoundError: No module named 'scipy'
```

```bash
pip install scipy
```

### CUDA unavailable

The pipeline defaults to `cuda:0`. If no GPU is available, training will fall back to CPU with a warning. CPU training is 10-50x slower and not practical for full runs. Use the smoke test (`smoke_200.yaml` with `amp: false`) for CPU-only validation.

### No metrics.json found

If `eval/metrics.json` is missing after running, check the eval log:

```bash
cat runs/nist_ir_ablation/runs/<model>/seed_<seed>/logs/eval_stdout.log
```

Common causes: missing checkpoint, decoding failure, or CUDA out-of-memory.

### Slurm array job failed

Check the Slurm output:

```bash
cat slurm_logs/nist_ir_ablation_<jobid>_<taskid>.out
```

Resubmit a single failed task using `run_matrix.py`:

```bash
python experiments/nist_ir_ablation/scripts/run_matrix.py \
  --output runs/nist_ir_ablation \
  --config experiments/nist_ir_ablation/configs/nist_ir_base.yaml \
  --model E0_atom_nope --seed 42 \
  --run_preprocess (if needed) --run_train --run_eval --aggregate \
  --skip_preprocess_if_exists
```

### How to rerun one model/seed

Use `run_matrix.py` in single-run mode:

```bash
python experiments/nist_ir_ablation/scripts/run_matrix.py \
  --output runs/nist_ir_ablation \
  --config experiments/nist_ir_ablation/configs/nist_ir_base.yaml \
  --model E1_atom_fourier --seed 2025 \
  --run_train --run_eval --skip_preprocess_if_exists
```

Or run the underlying scripts directly:

```bash
# Train
python experiments/nist_ir_ablation/scripts/train_one.py \
  --config experiments/nist_ir_ablation/configs/E1_atom_fourier.yaml \
  --processed_dir runs/nist_ir_ablation/processed \
  --output runs/nist_ir_ablation/runs/E1_atom_fourier/seed_2025 \
  --seed 2025 \
  --base_config experiments/nist_ir_ablation/configs/nist_ir_base.yaml

# Evaluate
python experiments/nist_ir_ablation/scripts/evaluate_one.py \
  --config experiments/nist_ir_ablation/configs/E1_atom_fourier.yaml \
  --processed_dir runs/nist_ir_ablation/processed \
  --checkpoint runs/nist_ir_ablation/runs/E1_atom_fourier/seed_2025/checkpoints/best.pt \
  --output runs/nist_ir_ablation/runs/E1_atom_fourier/seed_2025/eval \
  --seed 2025 \
  --base_config experiments/nist_ir_ablation/configs/nist_ir_base.yaml
```

---

## Requirements

- **Python 3.9+**
- **PyTorch 2.0+** (CUDA 11.8 recommended)
- **RDKit** (2022.09+)
- **scipy** (for cubic interpolation)
- **numpy**
- **pandas** (for CSV output)
- **pyyaml** (for config files)

Install core dependencies:

```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install rdkit-pypi scipy numpy pandas pyyaml
```

---

## Implementation Status

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | JSONL data loading, validation, SMILES canonicalization | Complete |
| Phase 2 | Train-one: Transformer training with teacher forcing | Complete |
| Phase 3 | Evaluate-one: Beam search decoding, Top-k accuracy, metrics | Complete |
| Phase 4 | Orchestration: `run_matrix.py`, `aggregate_results.py`, `make_report.py`, shell wrappers, Slurm array | Complete |

Phases 1-3 implement the core data/model/training/evaluation pipeline. Phase 4 provides the orchestration, aggregation, and reporting layers that compose the ablation experiment.
