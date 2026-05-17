# NIST IR 2x2 Ablation Package — Design Specification (Revised v2)

## 1. Overview

A self-contained experimental ablation package under `experiments/nist_ir_ablation/` that evaluates the contribution of two independent variables on SMILES prediction from experimental IR spectra:

| Factor | Levels |
|---|---|
| **Spectral encoding** | `nope` (intensity-only) vs `fourier` (intensity + Fourier wavenumber features) |
| **SMILES tokenization** | `atom` (atom-level tokenizer) vs `spe` (atom-level + BPE subword merges) |

Four conditions (`E0_atom_nope`, `E1_atom_fourier`, `E2_spe_nope`, `E3_spe_fourier`) × 3 seeds (42, 2025, 3407) = **12 runs** full; 4 runs (1 seed) smoke.

The package imports `src/model.py`, `src/util.py`, and `src/search_methods.py` from the existing TranSpec pipeline without modifying legacy code. A local decoder wrapper in `transpec_ext/decoding.py` adapts the search algorithm for configurable max_len, tokenizer-agnostic decode, and fair Top-k evaluation.

**Data note:** Current `IR_nist.jsonl` contains the full dataset. A 200-record subset (`IR_nist_200.jsonl`) is used for smoke testing. The parser handles variable wavenumber ranges (observed: ~630–4835 cm⁻¹) and point counts (observed: 2022–4872). Only ~1/200 records covers the full 552–3844 target range; out-of-range values fill to 0. Extra fields (temperature, pressure, condition, type, source) are stored in `raw` metadata.

## 2. Directory Layout

```
experiments/nist_ir_ablation/
├── README.md
├── configs/
│   ├── nist_ir_base.yaml           # shared defaults (full experiment)
│   ├── smoke_200.yaml              # overrides (smaller model, 2 epochs)
│   ├── E0_atom_nope.yaml           # condition-specific overrides
│   ├── E1_atom_fourier.yaml
│   ├── E2_spe_nope.yaml
│   └── E3_spe_fourier.yaml
├── transpec_ext/
│   ├── __init__.py
│   ├── data_jsonl.py               # IrSpectrumRecord, read_jsonl, deduplicate
│   ├── spectrum_preprocess.py      # resample, clip_negative, normalize, validate
│   ├── spectrum_encoding.py        # FourierWavenumberEncoding (nn.Module)
│   ├── vocab.py                    # special token constants (PAD/BOS/EOS/UNK)
│   ├── tokenizer_atom.py           # AtomTokenizer (BaseSmilesTokenizer)
│   ├── tokenizer_spe.py            # SPETokenizer (BPE on atom-tokenized seqs)
│   ├── dataset.py                  # NistIRAblationDataset, ablation_collate_fn
│   ├── model_factory.py            # AblationModel wrapper, build_model()
│   ├── decoding.py                 # threshold_value_search_ablation, decode_ablation
│   ├── metrics.py                  # Top-k accuracy, invalid decode rate
│   └── io_utils.py                 # save/load helpers, path construction
├── scripts/
│   ├── prepare_jsonl.py            # JSONL → processed data + tokenizer artifacts
│   ├── train_one.py                # Single condition+seed training
│   ├── evaluate_one.py             # Single condition+seed evaluation
│   ├── run_matrix.py               # Shared orchestration (prepare|train|evaluate|aggregate)
│   ├── aggregate_results.py        # 12 runs → summary CSVs + ablation effects
│   └── make_report.py              # Summary → report.md + report.html
├── tests/
│   ├── test_jsonl_reader.py
│   ├── test_spectrum_preprocess.py
│   ├── test_tokenizer_atom.py
│   ├── test_tokenizer_spe.py
│   ├── test_fourier_encoding.py
│   └── test_decoding.py
├── run_smoke_200.sh                # 4 conditions × 1 seed, small model, 2 epochs
├── run_full_12.sh                  # 12 runs, serial
├── run_full_12.slurm               # Slurm array job (requires pre-run prepare)
└── requirements_phase1.txt
```

## 3. Data Flow

```
IR_nist.jsonl (or IR_nist_200.jsonl)
  │
  └─ prepare_jsonl.py ──────────────────────────────────────────────────┐
       │  1. read_jsonl → valid + invalid records                         │
       │  2. canonicalize SMILES via RDKit                                │
       │  3. preprocess spectra (resample → clip_negative → normalize)    │
       │  4. remove failed (logged), create clean_records                 │
       │  5. fixed 80/10/10 split (split_seed=1234)                      │
       │  6. save artifacts to processed/common/                          │
       │  7. fit atom tokenizer on train SMILES only → save atom/         │
       │  8. fit SPE tokenizer on atom-tokenized train → save spe/        │
       │  9. encode split → atom/train.pt, atom/valid.pt, atom/test.pt   │
       │ 10. encode split → spe/train.pt, spe/valid.pt, spe/test.pt      │
       │ 11. save preprocess_manifest.json                                │
       └──────────────────────────────────────────────────────────────────┘

       ┌──────────────────────── 12× ──────────────────────────────┐
       │  train_one.py                                              │
       │    → load processed data + tokenizer                       │
       │    → build_model(config, tokenizer)                        │
       │    → train loop (config-driven hyperparams)                │
       │    → save checkpoints + logs + model_summary.txt           │
       │    → record best_epoch, best_valid_loss, train_time_sec    │
       │                                                             │
       └→ evaluate_one.py                                           │
           → load test data + tokenizer + checkpoint                 │
           → decode_ablation() → raw ranked candidates               │
           → canonicalize, compute Top-1/3/5/10 (raw ranking)        │
           → save metrics.json + predictions_topk.csv + candidates   │
           → record eval_time_sec                                    │
       └─────────────────────────────────────────────────────────────┘

       ┌─────────────────────────────────────────────────────────────┐
       │  aggregate_results.py                                       │
       │    → all_runs.csv (12 rows)                                 │
       │    → summary_mean_std.csv (4 rows, mean ± std)             │
       │    → ablation_effects.json (delta_pe, delta_spe,            │
       │                              delta_both, interaction)       │
       │                                                             │
       └→ make_report.py                                            │
           → report.md + report.html                                 │
       └─────────────────────────────────────────────────────────────┘
```

## 4. Output Directory Layout

The `--output` flag sets the experiment root. Paths derived automatically:

```
--output runs/nist_ir_ablation

  → processed_dir = runs/nist_ir_ablation/processed/
  → run_dir       = runs/nist_ir_ablation/runs/
  → summary_dir   = runs/nist_ir_ablation/summary/
```

All paths can be overridden explicitly (`--processed_dir`, `--run_dir`, `--summary_dir`), but the one-click scripts use the derived convention for portability. The entire output directory can be moved between machines.

### 4.1 Processed Dataset Structure

```
processed/
  common/
    records.pt                # canonical SMILES, raw SMILES, spectrum tensor (N, 3000), idx, metadata
    split_indices.json        # fixed split (split_seed=1234)
    preprocess_manifest.json  # runtime metadata
    smiles_stats.json         # max/p95/p99 token lengths, num_truncated
    spectrum_stats.json       # raw spectrum statistics
    invalid_records.jsonl     # skipped records with reason
  atom/
    train.pt                  # atom-tokenized target IDs (including BOS + tokens + EOS)
    valid.pt
    test.pt
    atom_vocab.json           # atom tokenizer vocabulary
  spe/
    train.pt                  # SPE-tokenized target IDs (including BOS + tokens + EOS)
    valid.pt
    test.pt
    spe_vocab.json            # SPE tokenizer vocabulary
    spe_merges.txt            # SPE merge rules
```

### 4.2 Run Directory Structure

```
runs/{model_id}/seed_{seed}/
├── config.resolved.yaml
├── preprocess_manifest.json
├── model_summary.txt         # includes total_params, trainable_params
├── checkpoints/
│   ├── best.pt               # model + optimizer + scheduler + metadata
│   └── last.pt
├── logs/
│   ├── train_log.csv
│   ├── valid_log.csv
│   └── stdout.log
├── eval/
│   ├── metrics.json          # model_id, seed, num_test, top1/3/5/10, best_epoch,
│   │                         # best_valid_loss, train_time_sec, eval_time_sec,
│   │                         # total_params, trainable_params
│   ├── predictions_topk.csv
│   ├── candidates.jsonl
│   ├── invalid_decodes.jsonl
│   └── test_examples_debug.csv
└── artifacts/
    ├── atom_vocab.json
    ├── spe_vocab.json         # only for spe conditions
    └── spe_merges.txt         # only for spe conditions
```

### 4.3 Summary Directory Structure

```
summary/
├── all_runs.csv
├── summary_mean_std.csv
├── ablation_effects.json
├── missing_runs.json          # if any
├── report.md
└── report.html
```

### 4.4 Examples

Smoke test:
```bash
bash experiments/nist_ir_ablation/run_smoke_200.sh \
  --input data/nist_ir/raw/IR_nist_200.jsonl \
  --output runs/nist_ir_ablation_smoke
# → runs/nist_ir_ablation_smoke/{processed,runs,summary}/
```

Full run:
```bash
bash experiments/nist_ir_ablation/run_full_12.sh \
  --input data/nist_ir/raw/IR_nist.jsonl \
  --output runs/nist_ir_ablation
# → runs/nist_ir_ablation/{processed,runs,summary}/
```

## 5. Component Specifications

### 5.1 Data Layer (`transpec_ext/data_jsonl.py`)

```python
@dataclass
class IrSpectrumRecord:
    idx: int
    smiles: str
    wavenumbers: List[float]
    intensities: List[float]
    condition: Optional[str] = None
    source: Optional[str] = None
    raw: Optional[dict] = None

# Functions
read_jsonl(path: str, min_points: int = 100) -> Tuple[List[IrSpectrumRecord], List[dict]]
    # Line-by-line. Invalid entries collected with line_number + reason.
    # Required: "smiles", "value.x", "value.y", same-length, >=min_points.
    # Additional fields (temperature, pressure, condition, type, source) stored in raw.

deduplicate_smiles(records: List[IrSpectrumRecord]) -> List[IrSpectrumRecord]
    # Optional. NOT enabled by default.
```

### 5.2 Spectrum Preprocessing (`transpec_ext/spectrum_preprocess.py`)

```python
resample_spectrum(wavenumbers, intensities, target_low=552.0, target_high=3844.0, n_points=3000)
    → Tuple[np.ndarray, np.ndarray]  # (grid, spectrum)
    # Sort ascending, merge duplicate x, cubic interpolation, fill_value=0.0
    # Linear interpolation fallback if <4 unique points.
    # Out-of-range values → 0.0 (important: many records don't cover full range)

normalize_spectrum(spectrum, mode="max") → np.ndarray
    # "max": divide by max positive intensity; zero-safe.
    # "max_100": MinMaxScaler → [0,100] (optional compat mode).

preprocess_record(record, n_points=3000, low=552.0, high=3844.0,
                  normalize="max", clip_negative=True,
                  filter_all_zero=False) → Optional[np.ndarray]
    # resample → clip_negative → normalize → validate → return [3000]

validate_spectrum(spectrum, expected_len=3000) → Tuple[bool, str]
    # Checks: shape, no NaN, no inf, finite.
```

### 5.3 Fourier Wavenumber Encoding (`transpec_ext/spectrum_encoding.py`)

```python
class FourierWavenumberEncoding(nn.Module):
    """
    Fourier encoding of the wavenumber position axis.
    Only modifies spectral input embedding — never touches attention,
    masks, or decoder positional encoding.

    Input:  (B, 1, N) — intensity only
    Output: (B, 1+2*L, N) — intensity + sin/cos features

    fourier_basis buffer: [1, 2*L, N]
      shape is [1, 2*L, N] so expand(B, -1, -1) avoids recomputation.

    Normalized wavenumber: p = (nu - nu_min) / (nu_max - nu_min)
    Features: sin(2π · 2^f · p), cos(2π · 2^f · p) for f = 0..L-1

    Equivalent to: h_i = W_I * I_i + W_nu * gamma(nu_i)
    """

    def __init__(self, num_frequencies, nu_min, nu_max, n_points):
        # Build grid from nu_min..nu_max, n_points
        # Compute sin/cos features for all frequencies
        # Register as buffer: [1, 2*num_frequencies, n_points]

    def forward(self, x):
        # x: (B, 1, N); assert N == self.n_points
        # basis = self.fourier_basis.expand(B, -1, -1)  → (B, 2L, N)
        # return cat(x, basis, dim=1)                    → (B, 1+2L, N)
```

Config driven: `full` uses `num_frequencies=32` (→ 65 channels), `smoke` uses `16` (→ 33 channels).

**CNN compatibility verified:** `src/model.py`'s `CNN.__init__` (line 127) accepts `input_channels` and uses it in the first `nn.Conv1d` layer. Passing `input_channels=65` (Fourier) or `input_channels=1` (no Fourier) works without modifying legacy code.

### 5.4 Vocabulary (`transpec_ext/vocab.py`)

```python
PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<PAD>", "<BOS>", "<EOS>", "<UNK>"
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

SPECIAL_TOKENS = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
FIRST_REAL_TOKEN_ID = 4
```

### 5.5 Tokenizer Interface

```python
class BaseSmilesTokenizer(ABC):
    def fit(self, smiles_list: List[str]) -> BaseSmilesTokenizer: ...
    def encode(self, smiles: str, add_special_tokens=True) -> List[int]: ...
    def decode(self, token_ids: List[int], skip_special_tokens=True,
               stop_at_eos=True) -> str: ...
    def save(self, path: str): ...
    @classmethod
    def load(cls, path: str) -> BaseSmilesTokenizer: ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def pad_id(self) -> int: return PAD_ID
```

#### 5.5.1 AtomTokenizer (`transpec_ext/tokenizer_atom.py`)

- Regex: bracket expressions `\[[^\]]+\]` first, then two-char elements (`Br`, `Cl`, `Si`, etc.), then `%[0-9]{2,}`, then single-char atoms, digits, bonds/branches.
- Full-match assertion: concatenated token strings must equal original SMILES.
- Unknown tokens → `UNK_ID` with warning collection.
- Vocab IDs start at `FIRST_REAL_TOKEN_ID` (4).
- Tokenizer does NOT include PAD/BOS/EOS/UNK in atom-level vocab (those are in SPECIAL_TOKENS).

#### 5.5.2 SPETokenizer (`transpec_ext/tokenizer_spe.py`)

- BPE on atom-tokenized sequences; trained on **train SMILES only**.
- Internal keys: `"SPE::left_key@@right_key"` — avoids collisions with atom tokens.
- Surface: `surface(left) + surface(right)` — correct SMILES substring.
- `min_frequency=2` (stop early if no pair qualifies).
- `num_merges` from config (default 100).
- `decode()` maps SPE token IDs to surfaces, never exposes internal keys.
- If a merge produces a non-SMILES substring (should not happen with atom-tokenized input), retains it as-is.

### 5.6 Dataset (`transpec_ext/dataset.py`)

```python
class NistIRAblationDataset(Dataset):
    def __getitem__(self, i) -> dict:
        return {
            "idx": int,
            "spectrum": FloatTensor [3000],
            "target_ids": LongTensor [seq_len],   # includes BOS + tokens + EOS
            "canonical_smiles": str,
            "raw_smiles": str,
        }

ablation_collate_fn(batch) -> dict:
    # Pads target_ids to uniform length
    # Generates decoder_input = target_ids[:, :-1]
    # Generates labels = target_ids[:, 1:]
    # Generates tgt_mask (batch_first compatible)
    # Generates tgt_padding_mask from decoder_input
    # Returns spectrum: (B, 1, 3000)
```

### 5.7 Model Factory (`transpec_ext/model_factory.py`)

```python
class AblationModel(nn.Module):
    """Optional Fourier encoding → core Model. Never modifies attention."""
    def __init__(self, core_model, fourier_encoding=None): ...
    def forward(self, en, de_1, tgt_mask, tgt_key_padding_mask):
        if self.fourier_encoding is not None:
            en = self.fourier_encoding(en)     # (B,1,N) → (B,1+2L,N)
        return self.core(en, de_1, tgt_mask, tgt_key_padding_mask)

build_model(config: dict, tokenizer) -> AblationModel
    # All hyperparams FROM config:
    #   d_model, nhead, num_encoder_layers, num_decoder_layers,
    #   dim_feedforward, dropout, max_len, label_smoothing, etc.
    # input_channels = 1 + 2*num_frequencies if fourier else 1
    # Creates Model(use_cnn=True, use_mlp=False, input_channels=...)
    # Saves model_summary.txt with total + trainable parameter counts
```

#### Default Hyperparameters

| Parameter | Full (5624) | Smoke (200) |
|---|---|---|
| d_model | 256 | 64 |
| nhead | 8 | 4 |
| num_encoder_layers | 4 | 2 |
| num_decoder_layers | 4 | 2 |
| dim_feedforward | 1024 | 256 |
| dropout | 0.2 | 0.1 |
| max_len | 256 | 256 |
| fourier.num_frequencies | 32 | 16 |
| label_smoothing | 0.05 | 0.0 |
| optimizer.lr | 2e-4 | 2e-4 |
| optimizer.weight_decay | 1e-4 | 1e-4 |
| optimizer.betas | (0.9, 0.99) | (0.9, 0.99) |
| grad_clip | 1.0 | 1.0 |

### 5.8 Decoding (`transpec_ext/decoding.py`)

```python
def threshold_value_search_ablation(
    model: Model,                # core_model (not AblationModel wrapper)
    encoder_input: torch.Tensor, # already Fourier-augmented if applicable
    tokenizer: BaseSmilesTokenizer,
    threshold_value: float = 0.01,
    candidate_limit: int = 500,
    max_len: int = 256,          # configurable (not hardcoded 99)
    use_cnn: bool = True,
    use_mlp: bool = False,
    input_channels: int = 1,
) -> Tuple[List[str], List[float], List[List[int]]]:
    """
    Local adaptation of src/search_methods.py threshold_value_search.

    Key differences:
    - max_len from config (not 99)
    - Uses tokenizer.decode() for SMILES reconstruction, not dic.get()
    - BOS/EOS/PAD from constants (PAD_ID=0, BOS_ID=1, EOS_ID=2)
    - Returns (surface_smiles, probabilities, token_id_sequences)
    """

def decode_ablation(
    ablation_model: AblationModel,
    spectrum: torch.Tensor,   # (1, 1, 3000)
    tokenizer: BaseSmilesTokenizer,
    **kwargs
) -> Tuple[List[str], List[float], List[List[int]]]:
    """Applies Fourier encoding (if present) exactly once, then delegates."""
    # Fourier applied here, NEVER inside threshold_value_search_ablation
```

**Top-k fairness rules:**
- Threshold, candidate_limit, max_len, ranking rule, invalid handling IDENTICAL across all 4 conditions.
- If threshold decoding produces <10 candidates: missing ranks stay empty, count as misses.
- No backfill with filtered/repaired/generated candidates.
- No filtering/repairing/canonicalizing before ranking.
- Canonicalization only for correctness comparison after ranking.

### 5.9 Training (`scripts/train_one.py`)

```
usage: train_one.py --processed_dir DIR --run_dir DIR --model_id ID
                    --seed SEED --config PATH [--force] [--resume]
```

Key behaviors:
- Loads processed data + tokenizer artifacts (never refits tokenizer)
- Config-driven: optimizer, scheduler (reduce_on_plateau), AMP, gradient clipping
- `optimizer.zero_grad(set_to_none=True)` before each step
- `clip_grad_norm` (default 1.0)
- `torch.cuda.amp.GradScaler(enabled=use_amp)`
- Validation mirrors training loss computation
- Scheduler stepped per epoch (reduce_on_plateau, factor=0.5, patience=5)
- Checkpoints: `best.pt` + `last.pt` (includes model, optimizer, scheduler, metadata with best_epoch, best_valid_loss)
- Records: best_epoch, best_valid_loss, train_time_sec
- Idempotent: skips if `best.pt` exists (unless `--force`)
- Reproducibility: config.resolved.yaml, git hash, seed, device info saved

### 5.10 Evaluation (`scripts/evaluate_one.py`)

```
usage: evaluate_one.py --processed_dir DIR --run_dir DIR --model_id ID
                       --seed SEED --config PATH [--force]
```

Key behaviors:
- Loads test data + saved tokenizer (never refits)
- Loads `best.pt` checkpoint
- For each test sample: `decode_ablation()` → raw ranked candidates + scores + token IDs
- RDKit canonicalization: each candidate once, cached
- **Top-k from raw ranked candidates** (invalid SMILES stay in rank, count as misses)
- Duplicate canonical SMILES: first occurrence counts, no reordering
- Records: eval_time_sec
- Writes `eval/metrics.json`, `eval/predictions_topk.csv`, `eval/candidates.jsonl`, etc.

### 5.11 metrics.json Structure

```json
{
    "model_id": "E0_atom_nope",
    "seed": 42,
    "num_test": 562,
    "top1": 0.35,
    "top3": 0.52,
    "top5": 0.61,
    "top10": 0.73,
    "best_epoch": 127,
    "best_valid_loss": 0.423,
    "train_time_sec": 1842.5,
    "eval_time_sec": 312.7,
    "total_params": 12345678,
    "trainable_params": 12345678
}
```

### 5.12 predictions_topk.csv Structure

```
idx,label_smiles,pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7,pred_8,pred_9,pred_10,hit_top1,hit_top3,hit_top5,hit_top10
0,CCCO,CCCO,CCN,CCO,,,C=O,,,,TRUE,TRUE,TRUE,TRUE
...
```

### 5.13 Aggregation (`scripts/aggregate_results.py`)

```
usage: aggregate_results.py --run_dir DIR --summary_dir DIR
                            [--strict true] [--conditions E0 E1 E2 E3]
```

- Lists expected 12 runs, checks existence
- Missing runs → `missing_runs.json` + warning
- Computes mean ± std per condition (sample std, ddof=1)
- Computes ablation effects from means:
  - `delta_pe = E1 - E0`
  - `delta_spe = E2 - E0`
  - `delta_both = E3 - E0`
  - `interaction = (E3 - E2) - (E1 - E0)`
- Outputs: `all_runs.csv`, `summary_mean_std.csv`, `ablation_effects.json`

### 5.14 summary_mean_std.csv Structure

```
model_id,top1_mean,top1_std,top3_mean,top3_std,top5_mean,top5_std,top10_mean,top10_std,total_params_mean
E0_atom_nope,0.350,0.012,0.520,0.015,0.610,0.011,0.730,0.009,12345678
...
```

### 5.15 ablation_effects.json Structure

```json
{
    "top1": {
        "delta_pe": 0.023,
        "delta_spe": -0.005,
        "delta_both": 0.015,
        "interaction": -0.013
    },
    "top3": { ... },
    "top5": { ... },
    "top10": { ... }
}
```

### 5.16 Report (`scripts/make_report.py`)

- Reads aggregate outputs
- Writes `report.md` and `report.html`
- Includes:
  1. Dataset summary (total records, split sizes, spectrum range, SMILES length stats)
  2. Split summary (train/valid/test counts)
  3. Model matrix (conditions table)
  4. Parameter count summary (per condition)
  5. Main result table (mean ± std for Top-1/3/5/10)
  6. Ablation effect table
  7. Short preliminary conclusion template:
     - Fourier PE: positive / neutral / negative
     - SPE: positive / neutral / negative
     - Combined model: best / not best
  8. Exact commands used to reproduce the smoke test and full experiment
- Reports trends as directional, not statistically significant (only 3 seeds)

### 5.17 Config Files (YAML)

**`nist_ir_base.yaml`** — shared defaults (full experiment):
```yaml
data:
  input: null
  processed_dir: null
  split_seed: 1234
  train_ratio: 0.8
  valid_ratio: 0.1
  test_ratio: 0.1

spectrum:
  target_len: 3000
  x_min: 552.0
  x_max: 3844.0
  clip_negative: true
  normalize: max
  filter_all_zero: false

fourier:
  num_frequencies: 32

model:
  d_model: 256
  nhead: 8
  num_encoder_layers: 4
  num_decoder_layers: 4
  dim_feedforward: 1024
  dropout: 0.2
  max_len: 256
  use_cnn: true
  use_mlp: false

training:
  epochs: 300
  batch_size: 32
  lr: 0.0002
  min_lr: 0.000001
  weight_decay: 0.0001
  betas: [0.9, 0.99]
  label_smoothing: 0.05
  grad_clip: 1.0
  amp: true
  num_workers: 4
  early_stopping_patience: 30

scheduler:
  type: reduce_on_plateau
  interval: epoch
  factor: 0.5
  patience: 5

decoding:
  threshold_value: 0.01
  candidate_limit: 500
  max_len: 256

tokenizer:
  max_len: 256

spe:
  vocab_size: 100
  min_frequency: 2
```

**`smoke_200.yaml`** — overrides:
```yaml
# Inherits nist_ir_base, then:
experiment:
  max_samples: 200
  seeds: [42]
  models: [E0_atom_nope, E1_atom_fourier, E2_spe_nope, E3_spe_fourier]

model:
  d_model: 64
  nhead: 4
  num_encoder_layers: 2
  num_decoder_layers: 2
  dim_feedforward: 256
  dropout: 0.1

fourier:
  num_frequencies: 16

training:
  epochs: 2
  batch_size: 8
  lr: 2.0e-4
  weight_decay: 1.0e-4
  early_stopping_patience: 2
  grad_clip: 1.0
  amp: false
  num_workers: 0
  label_smoothing: 0.0

decoding:
  max_len: 256
  threshold_value: 0.01
  candidate_limit: 100
```

**Condition configs** (minimal overrides):

`E0_atom_nope.yaml`:
```yaml
spectral_fourier_encoding: false
tokenizer:
  type: atom
```

`E1_atom_fourier.yaml`:
```yaml
spectral_fourier_encoding: true
tokenizer:
  type: atom
```

`E2_spe_nope.yaml`:
```yaml
spectral_fourier_encoding: false
tokenizer:
  type: spe
```

`E3_spe_fourier.yaml`:
```yaml
spectral_fourier_encoding: true
tokenizer:
  type: spe
```

### 5.18 Run Scripts

**`run_smoke_200.sh`:**
```bash
bash run_smoke_200.sh --input data/IR_nist_200.jsonl --output runs/nist_ir_ablation_smoke
```

**`run_full_12.sh`:**
```bash
bash run_full_12.sh --input data/IR_nist.jsonl --output runs/nist_ir_ablation
# To prepare only:
bash run_full_12.sh --prepare-only --input data/IR_nist.jsonl --output runs/nist_ir_ablation
```

**`run_full_12.slurm`** — requires pre-run data preparation:
```bash
# Step 1 (once, before array):
bash run_full_12.sh --prepare-only \
  --input /path/to/IR_nist.jsonl \
  --output /path/to/runs/nist_ir_ablation

# Step 2:
sbatch run_full_12.slurm

# Step 3 (after array completes):
python experiments/nist_ir_ablation/scripts/run_matrix.py \
  --action aggregate \
  --run_dir /path/to/runs/nist_ir_ablation/runs \
  --summary_dir /path/to/runs/nist_ir_ablation/summary \
  --strict true
```

**`run_matrix.py`** — shared orchestration:
```bash
python run_matrix.py --action prepare   --input PATH --processed_dir PATH
python run_matrix.py --action train     --processed_dir PATH --run_dir PATH --model_id ID --seed N --config PATH
python run_matrix.py --action evaluate  --processed_dir PATH --run_dir PATH --model_id ID --seed N --config PATH
python run_matrix.py --action aggregate --run_dir PATH --summary_dir PATH [--strict true]
```

### 5.19 Tests (CPU-compatible, pytest)

| File | Tests |
|---|---|
| `test_jsonl_reader.py` | valid/invalid/missing-field/dedup |
| `test_spectrum_preprocess.py` | resample/normalize/clip/validate/edge-cases/out-of-range |
| `test_tokenizer_atom.py` | fit/encode/decode/roundtrip/save-load/regex-full-match/Cl-Br/bracket/%10 |
| `test_tokenizer_spe.py` | merges/vocab-superset/internal-keys/surface/no-"SPE::"-in-output/save-load-determinism/train-only |
| `test_fourier_encoding.py` | output-shape/no-NaN/buffer-registration/batch-independence |
| `test_decoding.py` | max_len respected, tokenizer.decode used, SPE keys absent, raw ranking, <10-candidate gap handling |

## 6. Phase 1 — Data Preprocessing Details

### 6.1 JSONL Reading

- Read line by line.
- Each record: required fields `smiles`, `value.x`, `value.y`.
- Optional fields (stored in `raw`): `temperature`, `pressure`, `condition`, `type`, `source`.
- x and y must have equal length, >= min_points (default 100).
- Invalid JSON lines: collect with line_number + reason.
- Invalid SMILES (RDKit parse failure): collect, do not crash.

### 6.2 SMILES Canonicalization

- Use RDKit `Chem.MolFromSmiles` / `Chem.MolToSmiles`.
- Invalid SMILES → written to `invalid_records.jsonl` with reason.
- Canonical SMILES used as the label for training/evaluation.

### 6.3 Spectrum Preprocessing

- Resample to fixed 3000-point grid: 552.0–3844.0 cm⁻¹.
- Cubic interpolation; linear fallback if <4 unique points.
- Out-of-range wavenumbers → fill with 0.0.
- Clip negative intensities to 0.
- Normalize: divide by maximum positive intensity (zero-safe).
- Validate: shape, no NaN, no inf, finite.
- Do NOT apply shift/scale/quantize augmentation (would interfere with absolute wavenumber encoding).

### 6.4 SMILES Length Statistics

- Compute and save: max token length, p95 length, p99 length, number of truncated samples.
- If `num_truncated > 0`, print strong warning.
- Include in `smiles_stats.json` and `preprocess_manifest.json`.

### 6.5 Fixed Data Split

- 80% train, 10% valid, 10% test.
- split_seed = 1234 (fixed for reproducibility).
- `split_indices.json` saved once, reused by all 12 runs.

### 6.6 Tokenizer Training

- Atom tokenizer: fit on train SMILES only.
- SPE tokenizer: fit on atom-tokenized train sequences only.
- Valid/test SMILES never used for tokenizer training.

## 7. Condition Matrix Summary

| Condition | spectral_fourier_encoding | tokenizer_type | Decoder PE | spectral_input_channels |
|---|---|---|---|---|
| E0_atom_nope | false | atom | standard (sin/cos) | 1 |
| E1_atom_fourier | true | atom | standard (sin/cos) | 65 (1+2×32) |
| E2_spe_nope | false | spe | standard (sin/cos) | 1 |
| E3_spe_fourier | true | spe | standard (sin/cos) | 65 (1+2×32) |

## 8. Important Constraints

1. **Fixed split**: All 12 runs share `split_indices.json` (split_seed=1234). Seeds only affect model init, dropout, DataLoader shuffle.
2. **Tokenizer fit scope**: `fit()` called on **train SMILES only**. Valid/test call `encode()` only.
3. **No data augmentation**: No shift/scale/quantize. Would change wavenumber-position relationship.
4. **Attention unchanged**: Fourier encoding only modifies spectral input channels. `tgt_mask`, `src_mask`, decoder `PositionalEncoding`, and all attention mechanisms are identical across conditions.
5. **Fair Top-k**: Computed from raw ranked candidates. Invalid SMILES remain in ranking and count as misses. Candidates are not filtered before Top-k. Missing ranks (<10 candidates) stay empty, count as misses.
6. **Parameter counts recorded**: Every run saves `model_summary.txt`. The report includes parameter counts for interpreting ablation effects.
7. **Idempotent**: All scripts skip completed steps unless `--force` is set.
8. **Backward compatibility**: `src/model.py`, `src/util.py`, `src/search_methods.py` are NOT modified. The local `threshold_value_search_ablation` adapts the legacy algorithm with configurable max_len and tokenizer-agnostic decode.
9. **No SpecGNN, mass filtering, model fusion, or augmentation** in this experiment.
10. **Self-contained output**: `--output` flag controls the root of all derived paths. The entire `runs/` directory can be moved between machines.

## 9. Spec Self-Review

- **Placeholders**: None remaining.
- **Internal consistency**: Condition matrix, config inheritance, data flow, output paths consistent.
- **Scope**: Single ablation experiment. Fits in one repo under `experiments/`.
- **Ambiguity**: SPE = SMILES Pair Encoding (BPE subword). "nope" = no Fourier spectral encoding. Decoder always keeps standard positional encoding. Output dir paths derived from `--output`, not hardcoded.
- **Data compatibility**: Verified `src/model.py` accepts variable `input_channels` for CNN frontend. Verified JSONL format includes extra fields (temperature, pressure, condition, type, source) stored in `raw`.
