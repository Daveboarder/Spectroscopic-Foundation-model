# LIBS Foundation Model - Architecture Documentation

This document describes the architecture, masking strategies, and design decisions for the LIBS Foundation Model.

---

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Data Sources](#data-sources) — physics-version-2 generator, measured pipeline, splits, [line-token assets](#line-token-assets---line_embedding_config) and the CF dictionary
4. [Masking Strategies](#masking-strategies)
5. [Downstream Fine-Tuning](#downstream-fine-tuning) — incl. [calibration-free quantification](#calibration-free-quantification---task-cf_quantification)
6. [Training Configurations](#training-configurations)
7. [Evaluation Metrics](#evaluation-metrics)
8. [How BERT-style Masking Works for Continuous Spectra](#how-bert-style-masking-works-for-continuous-spectra)
9. [Design Decisions & Rationale](#design-decisions--rationale)
10. [Usage Examples](#usage-examples)

---

## Overview

The LIBS Foundation Model is a self-supervised transformer for Laser Induced Breakdown Spectroscopy. The model learns spectral representations through **Masked Intensity Prediction (MIP)** — predicting masked regions of the spectrum from context.

### Key Design Principles

1. **Peak-biased masking**: Ensures the model learns peak structure, not just noise
2. **Variable block sizes**: Prevents the model from exploiting fixed masking patterns
3. **Contiguous masking**: Forces global reasoning over local interpolation
4. **Scalable architecture**: Encoder size scales with `d_model` and `n_layers`; sequence length is either **17,428 bins** (intensity mode) or **`n_lines` line tokens** (line-token modes; 1,422 with `config/line_embedding.yaml`, 742 with `config/line_embedding_cf.yaml`), plus one CLS token
5. **Physics as the last layer**: for calibration-free quantification the encoder does not predict concentrations — a parameter-free Saha–Boltzmann solver does, fed by learned per-line weights and a learned initial plasma state (see [Downstream Fine-Tuning](#downstream-fine-tuning))

### Data and embedding modes

Training data is selected with `--libs_data_config`, which `build_dataset_from_config` (`data/libs_pipeline.py`) dispatches to one of three sources: the **physics-version-2 generator** (`config/libs_data.yaml`, `generation.plasma_model: two_zone` — Kirchhoff-consistent two-zone LTE radiative transfer, `data/two_zone_pipeline.py`; `one_zone` and `mixed` remain available for ablations), the **legacy physics-version-1 generator** (data configs without `plasma_model`, e.g. `config/libs_data_smoke.yaml`) or the **measured pipeline** (`config/libs_data_measured.yaml` with `source: measured`, handled by `data/measured_pipeline.py` — experimental Chameleon OptiCal JSON spectra interpolated onto the same 17,428-bin wavelength axis, unit-normalised and cached as `measured_cache_{md5}.h5`; build it ahead of time with `scripts/build_measured_dataset.py`). Without the flag both training scripts fall back to the legacy 5-class `SyntheticLIBSGenerator`.

| Embedding | `model.embedding_type` | Sequence | Pre-train target |
|-----------|------------------------|----------|------------------|
| **Bin (intensity)** | `intensity` (default) | ~17,428 bins + CLS | Masked bin intensities (MIP) |
| **Line-token** | `line_token` | `n_lines` + CLS (set by `line_dictionary.selection`; 1,422 lines with the default top-10%-per-element rule, 742 with the CF `cf_isolated` dictionary) | Masked Voigt features (`max_I`, `FWHM`) |
| **Line-token-linear** | `line_token_linear` | `n_lines` + CLS | Same MIP targets; reads `line_tokens_*.h5` |

Each mode supports two MIP losses, selected with `pretrain.loss`: `mse` (default — regression on the continuous targets) or `classification` (cross-entropy on targets discretized on the fly by `SpectroscopicDiscretizer`: 256 log-spaced intensity bins and, in line-token modes, 100 uniform FWHM bins, with `MaskedBinIntensityHead` / `MaskedLineFeatureHead` replacing the regression MIP head). `config_libs_token_linear_4090.yaml` and `config_libs_cf_4090.yaml` use `classification`; `config_libs_4090.yaml` uses `mse`.

Line-token modes require both `--line_embedding_config` (`config/line_embedding.yaml`, or `config/line_embedding_cf.yaml` for the calibration-free task) and `--libs_data_config`; the training scripts raise a `ValueError` if either is missing. Whenever `--line_embedding_config` is passed and `model.embedding_type` resolves to `intensity` (key absent **or** explicitly set to `intensity`), training switches to **`line_token_linear`** — bin-mode runs must therefore omit the flag. Checkpoints are **not interchangeable** between modes.

`--libs_data_config` overrides `data.n_bins` (intensity) or drives line-feature caches (line-token), and sets `model.max_seq_len` to `n_bins + 1` or `n_lines + 1`. Fine-tune also overrides `n_classes`, `n_elements`, and `n_concentration_bins` from the libs data config, and — when `--pretrain_run_dir` is given and that run has a `run_info.yaml` — `align_config_with_pretrain_run` first overwrites the `model:` keys (including `embedding_type`) with those from the pre-train run's `config.yaml` and inherits `--line_embedding_config` from its `run_info.yaml` if the flag is omitted (a warning is printed if both are given and differ).

---

## Model Architecture

### Transformer Encoder (high-level)

```
Intensity mode                    Line-token mode              Line-token-linear mode
──────────────                  ───────────────              ──────────────────────
Input [B, n_bins]               Input [B, L, 6] features       Input [B, L, 14] tokens
    │                               │                            │
    ▼                               ▼                            ▼
SpectralEmbedding               LineTokenEmbedding           LinearLineTokenEmbedding
MLP per bin                     runtime concat +             z-score + nn.Linear(14)
                                element/ion Embeddings       (pre-baked HDF5)
    │                               │                            │
    └───────────────────────────────┴────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────┐
│  + [CLS] Token (learned)                │
│  + Sinusoidal Positional Encoding       │
│  Seq len: n_bins+1  or  n_lines+1       │
│  (line mode: key_padding_mask invalid)  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Transformer Encoder Blocks (× N)       │
│  - Multi-Head Self-Attention            │
│  - Feed-Forward Network                 │
│  - Pre-LayerNorm + Residual Connections │
└─────────────────────────────────────────┘
    │
    ├────────────────────┐
    ▼                    ▼
[CLS] Embedding     Sequence Output
    │                    │
    │                    ├──────────────────────────────┐
    ▼                    ▼                              ▼
Classification /    MIP Prediction              CF (cf_quantification):
Regression /        Head (pre-training)         CFLineWeightHead (per line) + CFPlasmaInitHead (pooled)
Binned / Detection                              → parameter-free SahaBoltzmannLayer on the RAW tokens
Head (pooled:                                   → concentrations
cls | mean | cls_mean)
```

### Embedding Details

The embedding pipeline transforms raw intensities into the initial residual stream:

1. **Intensity projection** — a learned 2-layer MLP (`Linear(1→d_model/2) → GELU → Linear(d_model/2→d_model)`)
   projects each scalar intensity independently to a d_model-dimensional vector. The same
   weights are shared across all `n_bins` positions.

2. **Mask token replacement** (pretraining only) — masked positions have their projected
   vectors replaced with a single learned mask token (one d_model-dimensional vector, shared across all
   masked positions). See [How BERT-style Masking Works](#how-bert-style-masking-works-for-continuous-spectra)
   for the 80/10/10 replacement strategy.

3. **[CLS] token** — a learned d_model-dimensional vector prepended at position 0. It carries no spectral
   information initially; through attention across all layers it accumulates a global summary
   of the entire spectrum. Used for downstream classification/regression.

4. **Positional encoding** — fixed sinusoidal encoding added element-wise. This is the only
   thing differentiating masked positions from each other (they all share the same mask token).

5. **LayerNorm** — produces x₀, the initial state of the residual stream.

### Residual Stream Perspective (alternative view of the transformer)

The transformer can be understood as a single stream of vectors flowing straight from
embedding to output. Nothing overwrites it — each sub-layer reads from the stream,
computes a delta, and **adds** it back.

```
x₀  (embedding output — the initial stream)
 │
 │         ┌───────────────────────────┐
 ├────────►│  Self-Attention (layer 1) │
 │◄────────│                           │
 │  += δ   └───────────────────────────┘
 │
 │         ┌───────────────────────────┐
 ├────────►│  FFN (layer 1)            │
 │◄────────│                           │
 │  += δ   └───────────────────────────┘
 │
 │         ┌───────────────────────────┐
 ├────────►│  Self-Attention (layer 2) │
 │◄────────│                           │
 │  += δ   └───────────────────────────┘
 │
 │         ┌───────────────────────────┐
 ├────────►│  FFN (layer 2)            │
 │◄────────│                           │
 │  += δ   └───────────────────────────┘
 │
 ┆          ... layers 3–5 ...
 │
 │         ┌───────────────────────────┐
 ├────────►│  Self-Attention (layer 6) │
 │◄────────│                           │
 │  += δ   └───────────────────────────┘
 │
 │         ┌───────────────────────────┐
 ├────────►│  FFN (layer 6)            │
 │◄────────│                           │
 │  += δ   └───────────────────────────┘
 │
 ▼
x_final = x₀ + δ_attn1 + δ_ffn1 + ... + δ_ffn6
 │
 ▼
Final LayerNorm → output heads
```

**How to read this:**

- The straight vertical line **is** the residual stream — it is never overwritten, only added to.
- Self-attention reads all n_bins+1 positions, computes interactions between them, and writes a
  delta back. This is where tokens exchange information (masked positions gather context
  from unmasked neighbors).
- FFN processes each position independently — per-token "thinking" with no cross-talk.
- The stream can be tapped at **any intermediate point** and fed through the output head
  (MIP or classification) to get a valid prediction. Early taps give coarser results;
  later layers refine them. The final output is the sum of all deltas.
- At a masked position, x₀ has no intensity info — just mask token + positional encoding.
  The entire representation is built from context accumulated through attention deltas.
- At an unmasked position, x₀ already has the real intensity. Deltas enrich it with
  global spectral context.

### Model Configurations

| Config file | Embedding | d_model | n_layers | Use case |
|-------------|-----------|---------|----------|----------|
| `config_libs_smoke.yaml` | intensity | 32 | 1 | Pipeline wiring smoke |
| `config_libs_4090.yaml` | intensity | 128 | 4 | Bin embedding, RTX 4090 |
| `config_libs_a100.yaml` | intensity | 256 | 6 | Bin embedding, A100 |
| `config_libs_token.yaml` | line_token | 64 | 2 | Line-token smoke (+ `line_embedding_smoke.yaml`) |
| `config_libs_token_4090.yaml` | line_token | 128 | 4 | Line-token, RTX 4090 (+ `line_embedding.yaml`) |
| `config_libs_token_linear.yaml` | line_token_linear | 64 | 2 | Pre-baked tokens smoke |
| `config_libs_token_linear_4090.yaml` | line_token_linear | 128 | 6 | Pre-baked tokens, RTX 4090 (`pretrain.loss: classification`) |
| `config_libs_cf_smoke.yaml` | line_token_linear | 64 | 2 | CF task smoke: `config_libs_token_linear.yaml` + `finetune.cf` (+ `line_embedding_cf_smoke.yaml`, `libs_data_cf_smoke.yaml`) |
| `config_libs_cf_4090.yaml` | line_token_linear | 128 | 6 | CF task, RTX 4090: `config_libs_token_linear_4090.yaml` + `finetune.cf` (+ `line_embedding_cf.yaml`) |
| `config.yaml` | intensity | 256 | 6 | Legacy Gaussian-peak synthetic data (default `--config`) |
| `config_local.yaml` | intensity | 256 | 6 | Legacy synthetic data, 16 GB GPU |
| `config_a100.yaml` | intensity | 1024 | 24 | Legacy synthetic data, A100 (`run_pretrain.slurm`, `run_finetune.slurm`) |

At ~17k bins or ~0.7–1.4k lines (1,422 with the default top-10 % per-element selection in `config/line_embedding.yaml`, 742 with the CF dictionary of `config/line_embedding_cf.yaml`), `LIBSTransformer` uses PyTorch SDPA (`need_weights=False`) for tractable bf16 attention on consumer GPUs.

### Parameter Count Formula

```
params ≈ n_layers × (4 × d_model² + 2 × d_model × d_ff)
       + embeddings + heads
```

---

## Data Sources

### Physics-based pipeline (`data/libs_pipeline.py` + `config/libs_data.yaml`)

- `build_dataset_from_config(cfg)` is the single entry point. It dispatches on the data YAML: `source: measured` → the measured pipeline below; `generation.plasma_model` ∈ {`one_zone`, `two_zone`, `mixed`} → the physics-version-2 generator (`data/two_zone_pipeline.py`, below); key absent or `legacy` → the legacy physics-version-1 generator (`SyntheticLIBSDataset`, kept only so old caches still load).
- Shared inputs: line database `external_data/Source/LIBS_data.db` (air wavelengths; `QuantParam`, `E_ion`, `PartF_var`; stages I and II only; cleaned 2026-09-23: rows named `""`/`n`/`r` renamed to I/In/Ir — the iodine rows relabelled from stage II to I, their energies are those of neutral I — and the `X-II` groups, exact duplicates of their base element, deleted; 3,137 rare-earth lines added by `scripts/augment_line_db.py` (complete NIST ASD lines the DB lacked, plus Wisconsin laboratory Aki for Gd II, Sm II, Ce II, Eu II, Ho II and Dy I/II that NIST lists without Aki), with provenance in the table `QuantParam_source` (qp_id, source, accuracy) — existing NIST rows unchanged; `is_element_symbol` still guards element lists; H II gets U = 1 (bare proton) in `_load_partf`; spectra and line-dictionary cache keys gain `line_db` + `line_db_md5` (content hash), so they never collide with caches built from the legacy vacuum-wavelength `LIBS_data_vacuum.db` or from an older version of the DB), concentration ranges from `Samples_Fe_matrix.xlsx` (one sample type per row; `Uncertainties` sheet gives ± ranges, default ±1 %; element columns absent from the DB are dropped → 38 elements), and the 17,428-point wavelength axis from `external_data/Data/VASKUT K8.json` (two CCD segments).
- Per shot, concentrations are drawn independently per element in `(c − u, c + u)` and row-normalised; the element columns of the sample table are therefore **mass fractions** summing to 1. Downstream labels: these concentrations; classification labels via K-means on concentration vectors (`cluster_compositions`, `n_clusters` from `libs_data.yaml`).
- Materialises spectra once into `external_data/cache/synthetic_cache_{md5}.h5` (`measured_cache_{md5}.h5` for `source: measured`; `spectra` + `sample_table` group). `train_finetune.py` partitions that cache into train/val/test with `get_or_make_splits` and stores the indices in `splits_{md5}.json` (fractions from `downstream.splits`). `train_pretrain.py` does **not** read that file: it draws its own seeded 2-way train/val permutation over all cached spectra from `data.synthetic.val_fraction` (model config) and `--seed`, with no test hold-out. With the default `--seed 42` and `val_fraction: 0.15` (= `test_fraction`), the pre-train validation set happens to equal the fine-tune `random` test set and the pre-train training set contains the fine-tune validation set; this alignment is not enforced and breaks if either the seed, a fraction or the split strategy changes.
- Legacy generator (physics version 1; `ranges.te` uniform 6 500–11 000 K, `ranges.ne` log-uniform 1e17–5e17 cm⁻³, `number_density` 1e-4, `optical_path` 1.4e-4 cm): `create_spectra` scales each line by `Lp · (1 − e^{−τ})` with a fixed-width Voigt (σ 0.006 nm, γ 0.1 nm) and mass fractions fed straight into the intensity formula. Its source term `Lp ∝ λ⁻³ · N · e^{−ΔE/kT} · g_k/g_i` is not the Planck function, so the optically thin limit came out ∝ `λ · g_k² A_k / g_i` instead of `λ⁻¹ · A_k g_k e^{−E_k/kT}`. This is the reason for physics version 2; `config/libs_data_smoke.yaml` deliberately stays on the legacy generator to keep its cache key.

### Physics-version-2 generator (`data/two_zone_pipeline.py` + `data/plasma_physics.py`)

Selected by `generation.plasma_model: one_zone | two_zone | mixed`. Both `config/libs_data.yaml` and `config/libs_data_cf_smoke.yaml` use `two_zone`, so every shot is a hot core inside a cool shell; `mixed` draws two-zone shots with probability `two_zone_fraction` (0.7) and `one_zone` collapses the shell, both kept for ablations. Note that the CF loss terms `lambda_T` and `lambda_Ne` supervise T and Nₑ on one-zone shots only, so under `two_zone` they are inactive and the plasma-init head learns T/Nₑ only through the concentration and `lambda_Nl` terms. All line physics lives in `data/plasma_physics.py` (numpy, CGS), which is also the single source of truth for the line dictionary and the CF solver.

**Line physics (Kirchhoff-consistent).** For every DB line at a zone state (T, Nₑ) the integrated absorption per unit species density is `kt = λ⁴/(8πc) · A_ki g_k e^{−E_i/kT} (1 − e^{−ΔE/kT}) / U_s(T)` and the integrated emissivity `ε_int = (hc/4πλ) · A_ki g_k e^{−E_k/kT} · n_s/U_s(T)`, so that `ε/κ` equals the Planck source function `B_λ(T)` line by line (`line_source_function`, evaluated with ΔE = E_k − E_i so the identity holds exactly despite DB rounding). The optically thin limit of a line is therefore the textbook `(hc/4πλ) · A g_k e^{−E_k/kT} · n_s/U · l`, and the thick limit saturates at `B_λ(T)`. The species density `n_s = x_e · N · r_stage` uses the Saha split `S10 = n_II/n_I` (same algebra as the legacy code) and **number fractions** `x_e` obtained from the table's mass fractions with `data/atomic_data.py` (`mass_to_number_fractions`, IUPAC atomic weights; `number_to_mass_fractions` for the CF closure).

**Profiles and transfer.** Each line is spread with an area-normalised Voigt profile: thermal Doppler σ per element and zone (`doppler_sigma_nm`, from the atomic mass) plus one Lorentzian HWHM per zone, `gamma_stark`, as Stark proxy (Stark widths are not in the DB). On a fine grid (`fine_step_nm` 0.002, `±line_window_nm` 0.4 per line, widened up to 8× for saturated lines when `adaptive_window`), `κ_z(λ) = Σ_i κ_int,i φ_i(λ)`, `ε_z(λ) = Σ_i ε_int,i φ_i(λ)`, `S_z = ε_z/κ_z`, `τ_z(λ) = κ_z(λ) · l_z`. One-zone shots emit `S (1 − e^{−τ})`; two-zone shots use the three-slab recurrence outer → inner → outer (`two_zone_transfer`), which produces the self-reversed resonance lines of a hot core inside a cool shell. Transfer is solved **per element** and the emergent radiances are summed (lines of different elements do not absorb each other; lines of the same element are treated jointly, so blends within one element are exact). Lines below `min_relative_intensity` (1e-7) of the strongest thin line are dropped. The emergent radiance is convolved with the instrument kernel (`instrument.profile: gaussian`, FWHM 0.047 nm; `voigt` optional) **after** transfer, interpolated onto the spectrometer axis, optionally given a flat continuum pedestal and Gaussian noise (`augment`, both 0 by default) and unit-normalised (`unit_norm`, as the measured spectra).

**Zone parameter draws** (`generate_zone_sample_table`, one RNG stream per sample type):

| Quantity | Draw | Default range (`generation.zones`) |
|----------|------|-------------------------------------|
| `Te1` | uniform | 8 000–20 000 K (DB has no stage III) |
| `Ne1` | log-uniform | 5e16–3e18 cm⁻³ (covers the spark value 1.79e18) |
| `Te2` | `Te1 · U(te2_ratio)` | ratio 0.15–0.6 |
| `N1` | quasi-neutrality `N = Ne / Σ_e x_e r_II,e(T, Ne)` (`number_density: auto`; a float fixes it; `number_density_max` caps it) | — |
| `N2`, `Ne2` | `outer_density: isobaric` (default): pressure balance `N2 = N1 · Te1/Te2`, then `Ne2` from Saha equilibrium `Ne = N Σ_e x_e r_II,e(T, Ne)` solved by bisection in log10 Ne (`quasi_neutral` keeps the legacy rule `Ne2 = Ne1 · 10^U(ne2_log_ratio)` for ablations) | — |
| `l_inner`, `l_outer` | log-uniform (path length per traversal; the outer slab is crossed twice) | 0.003–0.05 cm, 2e-5–2e-3 cm |
| `gamma_stark1/2` | log-uniform, independently per zone | 0.003–0.03 nm |
| `plasma_model` | `two_zone` for every shot under `plasma_model: two_zone` (the default configs); under `mixed`, `two_zone` with probability `two_zone_fraction`, else `one_zone` (then `Te2 = Te1`, `Ne2 = Ne1`, `N2 = N1`, `l_outer = 0`) | `two_zone` (fraction 0.7 for `mixed`) |

**Calibration** (`scripts/calibrate_generator.py`, against measured PURE KFE spark spectra; the script never edits configs): the instrument FWHM 0.047 nm is the median of Gaussian(+linear baseline) fits to 15 isolated, weak, optically thin Fe lines; the path lengths come from matching two saturation statistics — the area ratio of the five strongest Fe I resonance lines (E_i < 0.2 eV) to five weak high-E_k lines (core column density `N·l` 5e15–3e16 cm⁻²) and of the resonance lines to strong lines with 0.8 < E_i < 1.6 eV, which only the cold shell absorbs (shell `N·l` 1e14–3e14 cm⁻²). With the quasi-neutral densities this gives the config ranges `l_inner_cm: [0.003, 0.05]` and `l_outer_cm: [2e-5, 2e-3]` (log-uniform because they span more than a decade).

**Sample table (contract C1).** Columns: `sample_type_id`, `sample_type_name`, `unique_id`, the 38 element mass fractions, the legacy aliases `Te`/`Ne` (= `Te1`/`Ne1`), and the zone columns `ZONE_COLUMNS` = `plasma_model`, `Te1`, `Ne1`, `Te2`, `Ne2`, `l_inner`, `l_outer`, `N1`, `N2`, `gamma_stark1`, `gamma_stark2` (metadata, never treated as elements). `extract_plasma_targets(sample_table)` turns them into the per-shot CF targets `Te` (K), `log10_Ne`, `log10_Nl` = log10(N1 · l_inner) (the column density that sets the inner-zone optical depth), `is_two_zone` and `has_plasma_labels` (1 only when the zone columns exist and are positive — 0 for measured spectra and legacy caches, every other entry then being 0). The cache key of `synthetic_cache_{md5}.h5` hashes every `generation:` knob plus `physics_version: 2` and the sample types (`TwoZoneSyntheticDataset.cache_key`), so the v2 cache never collides with a legacy one; the HDF5 layout is unchanged.

**Checks** (`scripts/check_two_zone_physics.py`, report + figures under `sanity_checks/two_zone_<timestamp>/`): (a) Kirchhoff `ε/κ = S` and thin-limit area = `ε_int · l`; (b) thick limit → `S` and → `B_λ(T)`; (c) two identical zones with `l_inner + 2 l_outer = l` reproduce one zone; (d) self-reversal of strong Fe I resonance lines for a hot core / cold shell; (e) round trip through `cf.solver_np` on a thin one-zone shot (T ±1 %, log10 Ne ±0.04, majors ±5 %); (f) fine-grid convergence (halving `fine_step_nm` changes the spectrum by < 1e-3); (g) end-to-end smoke dataset from `--libs_data_config` (shapes, contract-C1 columns, `extract_plasma_targets`, measured cache → `has_plasma_labels` 0, `measured_groups` / grouped splits).

### Measured-spectra pipeline (`data/measured_pipeline.py` + `config/libs_data_measured.yaml`)

- Selected by `source: measured` in the libs data config, so the same `--libs_data_config` flag drives both sources.
- Reads Chameleon/OptiCal analysis JSON files under `paths.measured_json_root` (`*_info.json` skipped), keeps the first run with `type == IncInAverage` (`run_policy: first_valid`), concatenates CCD ranges `[1, 2]` of integration phase 1, interpolates onto the `VASKUT K8.json` wavelength axis and applies `unit_norm`.
- Concentrations come from `Samples_Fe_matrix.xlsx`, matched by JSON filename stem and row-normalised to sum to 1 (same 38 elements); `Te`/`Ne` are stored as 0, so `has_plasma_labels` is 0 for every measured spectrum. `unique_id` is `{sample_type_id}_{INSTRUMENT}_R{run:02d}`, which is what the grouped splits parse.
- Cached to `external_data/cache/measured_cache_{md5}.h5` with the same HDF5 schema as `synthetic_cache_{md5}.h5`; the splits file and the line-token caches are keyed by the measured `cache_key` in the same way. The current cache holds 3,346 spectra from 121 sample types.
- `scripts/build_measured_dataset.py --libs_data_config config/libs_data_measured.yaml [--line_embedding_config ... --build_line_tokens] [--max_files N]` builds and validates the cache ahead of training.

### Train/val/test splits (`get_or_make_splits`, fine-tune only)

`downstream.splits.strategy` (overridable with `train_finetune.py --split_strategy`) selects the partition rule: `random` — per-row permutation, file `splits_{key}.json`; `group_sample` — whole groups together with `groups = sample_type_id`, so every shot of a physical sample lands in one partition; `group_instrument` — groups parsed from `unique_id` (`measured_groups(sample_table, by='instrument')`, e.g. `PURE_KFE_REMUS_9951601_R02` → `REMUS_9951601`). Grouped strategies write `splits_{key}_{strategy}.json`, assign shuffled groups greedily (test first until `test_fraction` of the rows is reached, then val, the rest train; at least one group per partition) and are the default for `libs_data_measured.yaml`, so the measured test set never leaks a sample seen in training. A cached split file is reused only if its row count still matches the cache. The plasma targets are split alongside the labels (`aux` in `generate_labeled_data`).

### Runtime overrides (`--libs_data_config`)

| Field | New value | Scripts |
|-------|-----------|---------|
| `data.n_bins` | length of wavelength array (intensity) or `n_lines` (line-token) | `train_pretrain.py`, `train_finetune.py` |
| `model.max_seq_len` | `n_bins + 1` (intensity) or `n_lines + 1` (line-token) | same |
| `data.n_classes` | `downstream.n_clusters` | `train_finetune.py` only |
| `data.n_elements` | count of `elements_to_predict`, or all sample-matrix element columns that also exist in the line DB (38 of the 60 columns in `Samples_Fe_matrix.xlsx`; the plasma-state columns are excluded) | `train_finetune.py` only |
| `data.n_concentration_bins` | `downstream.n_concentration_bins` | `train_finetune.py` only |

Fine-tune `run_info.yaml` records `libs_data_config`, `line_embedding_config` (null when the flag is absent), `line_features_path`, `line_tokens_path`, `spectra_cache_path` and `split_strategy`. Pre-train `run_info.yaml` records `embedding_type`, `pretrain_loss`, the `discretization` block, `n_lines`, `line_embedding_config`, `line_features_path` and `line_tokens_path`, but **not** `libs_data_config`.

### Line-token assets (`--line_embedding_config`)

Three HDF5 caches are built in sequence (additive — intermediate files are kept for debugging):

| Step | Module | Output | Role |
|------|--------|--------|------|
| 1 | `data/line_dictionary.py` | `line_dict_{hash}.h5` | Thin-limit theoretical intensities over a Te×Ne grid (`plasma_physics.thin_line_intensities`, grid maximum per line; `LINE_DICT_PHYSICS_VERSION = 2` is part of the hash, the legacy `N`/`C`/`l` keys are ignored), clipped to the VASKUT axis; `selection.mode` = `top_percent_per_element` (default: top 10 % per element, `min_keep: 10`), `threshold` (legacy) or `cf_isolated` (below). Datasets: `central_wavelength`, `theoretical_intensity`, `reference_intensity`, `isolation_score`, `forced`, `Te_opt`, `Ne_opt`, `Ei`, `Ek`, `gi`, `gk`, `Ak`, `element_id`, `ion_state_id`, `vocab/` |
| 2 | `data/line_features.py` | `line_features_{hash}.h5` | Per-spectrum Voigt fits in a `±window_nm` (0.3 nm) window: `[n_spectra, n_lines, 6]` (`max_I` = fitted **area**, `FWHM`, `R²`, `Δλ`, `RMSE`, `fit_valid` with `r2_min` 0.85). Options: `baseline: none \| linear` (Voigt + `b0 + b1·(λ − centre)`, R² judged on the baseline-subtracted data), `max_delta_nm` (fitted centre within ± of the DB wavelength) and `max_fwhm_nm` (reject broader fits); the legacy defaults keep the old cache hashes |
| 3 | `data/line_tokenization.py` | `line_tokens_{hash}.h5` | **Separated tokenization**: merged 14-feature tensor per line, raw values + `feature_mean`/`feature_std` attrs |

**Standalone build** (all three steps; each step is skipped on a cache hit). `scripts/build_line_tokens.py` loads the spectra with `build_dataset_from_config()` (generating the spectra cache if needed) and then calls `prepare_line_tokens_assets()` (`data/line_embedding_pipeline.py`), the same function the `line_token_linear` trainers call:

```bash
uv run python scripts/build_line_tokens.py \
  --libs_data_config config/libs_data.yaml \
  --line_embedding_config config/line_embedding.yaml   # or config/line_embedding_cf.yaml
```

Step 3 only, from existing step 1–2 files:

```bash
uv run python -m data.line_tokenization \
  --line_dict_path external_data/cache/line_dict_{hash}.h5 \
  --line_features_path external_data/cache/line_features_{hash}.h5
```

`line_token` mode builds only steps 1–2 via `prepare_line_token_assets()`; `line_tokens_{hash}.h5` is written only for `line_token_linear`.

#### CF dictionary (`selection.mode: cf_isolated`, `config/line_embedding_cf.yaml`)

> **Mineral fork:** `config/line_embedding_cf*.yaml` now take the targets from `external_data/Source/REE_minerals_oxides.xlsx`, treat air (`{N: 0.78, O: 0.21, Ar: 0.0093}`) as ambient, clip to the LIGHTIGO span 188.6–859.2 nm and force `cf/data/cf_mineral_lines.tsv` (140 air-wavelength analytical lines of the mineral, volatile, air and REE elements; `run_cf_classical.py --line_list cf_minerals`). The steel description below (54 CF-OES lines, Ar, 38 matrix elements) is the upstream setup.

The calibration-free task needs lines that belong unambiguously to one element. In this mode the candidates are restricted to `target_elements` (`from_sample_matrix`: the element columns of `Samples_Fe_matrix.xlsx` present in the DB, i.e. the 38 matrix elements) and every candidate *i* of element *e* is scored at a reference plasma (`reference_te` 10 000 K, `reference_ne` 1e17 cm⁻³):

```
interference_i    = Σ_{j ≠ i, |λ_j − λ_i| ≤ isolation_window_nm}  I_j · C_j^max
ratio_i           = interference_i / (I_i · C_e^typ)
isolated_i        = ratio_i ≤ max_interference_fraction
isolation_score_i = clip(1 − ratio_i, 0, 1)
```

The sum runs over all DB lines of every matrix element (including other lines of the candidate's own element), each weighted by that element's maximum matrix concentration expressed as a number fraction (`interferer_units: number`), plus `ambient_elements` (`{Ar: 1.0}`: the spark runs in argon, so Ar lines interfere at full weight although Ar is not in the matrix). `C_e^typ` is the candidate element's own maximum concentration floored at `typical_concentration_floor` (1e-4). Per element the isolated lines are ranked by intensity and the top `max_lines_per_species` (40) kept; if fewer than `min_lines_per_species` (20) are isolated, the best non-isolated lines fill up while keeping their low score. Rows of `force_include` (`cf/data/cf_oes_lines_54.tsv`, the 54 curated CF spark-OES lines: Fe 19, Ni 9, Cu 8, Mn 7, C 3, Cr 3, Si 3, Al 2) matched by (element, stage, |Δλ| ≤ 0.01 nm) are always kept and flagged `forced = 1`; the file's md5 is part of the cache hash. The current full dictionary has **742 lines** (20 per element for most elements, Fe 28, fewer where the DB has fewer candidates; 54 forced), of which only 64 pass the isolation threshold — the rest are fill lines whose `isolation_score` tells the CF solver how much to trust them. `line_embedding_cf_smoke.yaml` (8 / 4 lines per species) gives 187 lines. `line_features` for the CF configs uses `baseline: linear`, `max_delta_nm: 0.1`, `max_fwhm_nm: 0.2` and instrument-like initial widths (`gamma_init` = `sigma_init` = 0.02 nm).

#### `line_tokens_*.h5` layout

```
tokens          [n_spectra, n_lines, 14]  float32   # raw features (z-scored at train time)
fit_valid       [n_spectra, n_lines]      uint8     # 1 = successful Voigt fit
central_wavelength [n_lines]             float32   # for sinusoidal PE
attrs: n_spectra, n_lines, n_features, feature_names (JSON), feature_mean, feature_std,
       line_dict_hash, line_features_hash, line_dict_path, line_features_path, config_hash,
       dynamic_feature_indices, mip_target_indices
```

**14 channels (indices):** 0 λ, 1 Ei, 2 Ek, 3–6 log10(gi/gk/Ak/I_theory), 7 atomic_number (Z), 8 ion_binary (0=I, 1=II+), 9 max_intensity (= fitted Voigt **area**, the profile is area-normalised), 10 fwhm, 11 r2, 12 delta_lambda, 13 rmse. Channel 6 is the thin-limit integrated radiance per unit `x_e · N · l` from `plasma_physics.thin_line_intensities` (grid maximum over Te × Ne); the CF solver reads channels 0–9 directly.

#### Embedding at training time

| Mode | Reads | Embedding module |
|------|-------|------------------|
| `line_token` | `line_features_*.h5` per spectrum + `line_dict_*.h5` (static per-line buffers, loaded once at model init) | `LineTokenEmbedding` — 7 quantum scalars + `nn.Embedding` element/ion + 5 fit features → 2-layer MLP |
| `line_token_linear` | `line_tokens_*.h5` only | `LinearLineTokenEmbedding` — z-score using stored stats → `nn.Linear(14, d_model)` |

Invalid Voigt fits use `key_padding_mask` from the separate `fit_valid` dataset (not mixed into the 14 feature channels).

Pre-training masks a fraction of valid lines and predicts masked `max_intensity` and `fwhm`. The datasets are `MaskedLineTokenDataset` (`line_token`) and `MaskedLineTokensDataset` (`line_token_linear`), both defined in `data/dataset.py`; `PretrainDataModule.setup()` in `training/pretrain.py` picks one depending on whether `line_tokens_path` or `line_features_path` is set. Fine-tuning reads the same caches through `LineTokenLabeledDataset` / `LineTokensLabeledDataset`, which also emit the per-spectrum plasma targets (`aux_targets`) needed by the CF task.

---

## Masking Strategies

Masking below applies to **intensity (bin) mode**. Line-token modes mask entire line tokens and zero the `max_intensity` / `fwhm` channels in the input (indices 9–10 in `line_tokens_*.h5`, or channels 0–1 in `line_features_*.h5`).

The masking strategy is crucial for learning useful representations. We implement several options:

### 1. Random Scattered Masking (BERT-style)

```yaml
contiguous_masking: false
```

- Masks individual bins scattered randomly
- Simple but may allow local interpolation
- Not recommended for sparse spectral data

### 2. Contiguous Block Masking

```yaml
contiguous_masking: true
block_sizes: [50]  # Fixed size
```

- Masks contiguous regions of fixed size
- Forces the model to use global context
- Better than random for structured data

### 3. Variable Block Size Masking (Recommended)

```yaml
contiguous_masking: true
block_sizes: [25, 50, 100, 150]
```

- Block size randomly sampled from list
- Prevents model from exploiting fixed patterns
- Creates diverse training signal

### 4. Peak-Biased Masking (Recommended)

```yaml
contiguous_masking: true  # required: peak bias only applies to contiguous block masking
peak_bias_enabled: true
peak_bias_ratio: 0.5      # 50% of masks must cover peaks
peak_threshold: 0.2       # Intensity > 0.2 = peak region
```

**Why Peak-Biased?**

LIBS spectra are sparse — most bins are noise/baseline (~95%). Random masking primarily masks noise, making the task trivial. Peak-biased masking ensures:

- At least `peak_bias_ratio` of masked bins overlap with peaks
- Model must learn peak structure, not just "predict low values"
- More challenging and informative training signal

**Peak Detection Algorithm:**

```python
# Simple threshold-based detection
peak_mask = spectrum > peak_threshold

# Expand to include peak shoulders (±5 bins)
for each peak position:
    mark positions [peak-5 : peak+5] as peak region
```

### Masking Statistics

The training script logs masking statistics:

```
Masking Statistics (from 100 samples):
  Avg masked bins: 307.2
  Avg masked ratio: 15.00%
  Peak coverage: 52.3%      # % of masks on peaks
  Avg peak bins: 156.8      # Avg bins in peak regions (intensity > threshold, expanded by ±5-bin shoulders)
```

---

## Downstream Fine-Tuning

Fine-tuning attaches task-specific heads to the frozen or jointly trained `LIBSTransformer` encoder via `LIBSFinetuneModule` (`training/finetune.py`). Heads live in `models/heads.py`.

### Pooling (`--pool`)

The encoder returns `cls_embedding` and `sequence_embeddings`. Fine-tuning collapses them before the heads:

| Pool | Representation | Head input dim |
|------|----------------|----------------|
| `cls` | CLS token only | `d_model` |
| `mean` | Mean over sequence positions (bins or lines, excluding CLS; invalid-fit lines are excluded through `key_padding_mask`) | `d_model` |
| `cls_mean` | Concatenation of CLS and mean-pool | `2 × d_model` |

### Task modes (`--task`)

| Task | Head | Loss | Targets in batch |
|------|------|------|------------------|
| `classification` | `ClassificationHead` | Cross-entropy | `label` |
| `quantification` | `RegressionHead` (sigmoid) | MSE | `concentrations` |
| `quantification_binned` | `BinnedQuantificationHead` | Per-element CE on 1000 bins | `concentrations` (bins encoded on the fly) |
| `detection` | `DetectionHead` | Multi-label BCE-with-logits (per-element `pos_weight`) | `concentrations` (binarized on the fly as `concentration >= LOD`) |
| `cf_quantification` | `CFLineWeightHead` + `CFPlasmaInitHead` feeding the parameter-free `SahaBoltzmannLayer` (`cf/`) | Log-space concentration loss + plasma-state and weight regularisers (synthetic shots only) | `concentrations`, `Te`, `log10_Ne`, `log10_Nl`, `is_two_zone`, `has_plasma_labels` |
| `regression` | alias of `quantification` | MSE | `concentrations` |
| `both` | classification + regression | CE + MSE | `label` + `concentrations` |

`BinnedQuantificationHead` uses one small MLP branch per element, each outputting `n_concentration_bins` logits (default 1000). `concentration_to_bin` / `bin_to_concentration` in `models/heads.py` map between float targets in [0, 1] and bin indices.

`DetectionHead` outputs one presence logit per element (sigmoid ≥ 0.5 → present). Targets come from `concentration_to_presence` in `models/heads.py` (`concentration >= LOD`), with per-element limits of detection read from `--element_lod_config` (default `config/element_lod.yaml`: `limits_of_detection` per element, `default_lod: 1.0e-4` fallback, mass-fraction units); the BCE `pos_weight` is `n_absent / n_present` per element on the training split, clipped to [0.1, 100]. `detection` requires `--libs_data_config` so element names are known. It logs `det_loss`, `det_accuracy`, `det_precision`, `det_recall`, `det_f1` and `det_exact_match`; `run_info.yaml` stores `element_lod_config`, the resolved `element_lod` map and `test_results.detection` (macro / micro precision, recall and F1, `element_accuracy`, `exact_match`, per-element precision / recall / F1 / support).

### Calibration-free quantification (`--task cf_quantification`)

The CF task replaces the regression head by physics: concentrations are the output of a **parameter-free Saha–Boltzmann solver** (`cf/solver_np.py` reference in numpy, `cf/layer.py` its batched, differentiable float64 torch port; `scripts/check_cf_layer.py` asserts both agree to round-off, checks gradients, degenerate cases and the curve-of-growth table). The encoder only decides *which lines to trust* and *where the solver starts*. It requires the `line_token_linear` path (raw tokens from `line_tokens_*.h5`, ideally built with `config/line_embedding_cf.yaml`) and `--libs_data_config`.

**Calibration-free protocol.** The CF heads are trained on synthetic physics-v2 shots only — the loss is averaged over samples with `has_plasma_labels == 1`, and a batch without plasma labels contributes a zero (but differentiable) loss. Measured spectra never enter training; they are evaluation-only (`scripts/evaluate_cf.py`, zero-shot).

| Component | What it does |
|-----------|--------------|
| Inputs | Raw token channels 0–9 (λ, Eᵢ, Eₖ, log10 gᵢ, log10 gₖ, log10 Aₖ, log10 I_theory, Z, ion stage, fitted Voigt **area**) and `fit_valid`; the z-scored embedding is only used by the encoder |
| `CFLineWeightHead` (learned) | Per-line MLP (`d_model → 64 → 1`, `finetune.cf.weight_hidden`) on `sequence_embeddings`; `w = sigmoid(logit) · fit_valid` are the least-squares weights |
| `CFPlasmaInitHead` (learned) | MLP on the pooled representation giving the solver's start and prior centres, bounded by sigmoids: `T0` ∈ 5 000–25 000 K, `log10 Ne0` ∈ 15–19, `log10 N·l0` ∈ 13–19 (column density for the optical depths). Zero-initialised output layer, so every spectrum starts at 10 000 K / 10¹⁷ cm⁻³ / 10¹⁶ cm⁻² |
| Seed `--seed_binned_run_dir` | Frozen `quantification_binned` run on the **same** token cache (line-dictionary hash is checked; loaded from `best.ckpt`, kept in eval mode); its normalised prediction is the closure seed `C0` (`--cf_c0_source binned`; `uniform` or `truth` for debugging) |
| Seed `--seed_detection_run_dir` | Frozen `detection` run on the same cache; `presence ≥ 0.5` is mapped onto the lines through the atomic-number channel and multiplies the weights (`finetune.cf.presence_gate: true`); lines of non-target elements get weight 0 |
| `SahaBoltzmannLayer` | No trainable parameters (tables are buffers), float64 with autocast disabled; differentiable w.r.t. `w`, `T0`, `log10 Ne0`, `log10 N·l0` |

**Solver maths** (`cf/solver_np.py`, `cf/__init__.py`). For line *i* of element *e* in stage *z* ∈ {0, 1}:

```
y_i = ln(area_i · λ_i / (g_k A_k)) − z · ln F(T),      F(T) = 2 (2π m_e k T / h²)^{3/2}
y_i = q_e − (E_k,i + z · E_ion,e) · β − z · η,          β = 1/kT [eV⁻¹],  η = ln N_e
```

The unknowns θ = [q₁ … q_E, β, η] come from a weighted least squares with prior rows `prior_T·(β − β₀)²` and `prior_Ne·(η − η₀)²` (centres from the init head) plus a `ridge`; the T-dependence of F(T) is refreshed in a fixed-point loop of `n_iter` (default 3) solves. **Self-absorption** (`sa_correction`): after each solve the closure gives number fractions *x*, every line's centre optical depth is `τ₀,i = kt_i(T) · x_e · r_z(T, N_e) · (N·l) · φ_peak,i` (Doppler σ from the atomic mass, Lorentz HWHM `gamma_nm` = 0.01 nm), and the next solve uses `area_i / f(τ₀,i)` with the curve-of-growth factor *f* from the `CFTables` lookup (the first solve already uses τ₀ from the seed composition, `sa_seed_init`). **Closure**: `x_e ∝ U_I,e(T) · e^{q_e} · (1 + S10_e(T, N_e))`; elements without any used line take their share from `C0`; number → mass fractions via the atomic masses (`data/atomic_data.py`); elements below the per-element LOD (`config/element_lod.yaml`) are flagged `censored`. **Guards** (a spectroscopist's rules, no parameters): elements with fewer than `min_lines` (2) weighted lines are not solved by CF and fall back to the seed; between solves, lines whose residual deviates from the median by more than `reject_sigma` (3) robust sigmas (1.4826·MAD, floored at `reject_floor` 0.15 ln-units) are dropped and the gating re-applied. `CFTables` (`cf/tables.py`, `build_cf_tables(element_names, db_path, element_lod_config)`) holds E_ion, atomic masses, LODs, U_I/U_II on a 3 000–30 000 K grid and the curve-of-growth table.

**Loss** (`compute_cf_loss`, weights from `finetune.cf`):

```
L = L_conc + λ_T · ((T − Te)/Te)² [one-zone]  + λ_Ne · (log10 Ne − log10 Ne_true)² [one-zone]
           + λ_Nl · (log10 N·l0 − log10 N1·l_inner)² + λ_w · (mean_{fit_valid} w − 1)²
L_conc = mean_e[ m_e (ln(C_e+ε) − ln(C_true,e+ε))² ] + mean_e[ (1−m_e) relu(ln(C_e+ε) − ln LOD_e)² ],   m_e = 1[C_true,e ≥ LOD_e]
```

The T / Ne terms use only one-zone shots (a two-zone plasma has no single temperature); `λ_w` keeps the weight head from switching every line off. Metrics: see [Evaluation Metrics](#evaluation-metrics); `ModelCheckpoint` monitors `val/cf_log_rmse` (min).

**Pure-physics variant** (`--cf_pure_physics`): the learned heads are bypassed — `cf.classical.classical_weights` gives {0, 1} weights (fit valid, area > 0, R² ≥ 0.9, and isolation score ≥ 0.5 unless the line is force-included) and the solver starts from the defaults (10 000 K, 10¹⁷ cm⁻³, 10¹⁶ cm⁻²); seeds still apply. It is the zero-parameter reference that the learned run must beat. `scripts/run_cf_classical.py` runs the same numpy solver over a token cache without any encoder (`--line_list cf_oes54` restricts it to the 54 curated CF spark-OES lines in `cf/data/cf_oes_lines_54.tsv`; `all` uses every gated line).

`run_info.yaml` of a CF run gains a `cf` block (`seed_binned_run`, `seed_detection_run`, `pure_physics`, `cf_cfg` with the resolved solver settings, `line_dict_path`, `spectra_cache_path`, `split_strategy`, `c0_source`) plus `element_lod_config` / `element_lod`, and `test_results` carries `per_element` and `test_plasma_metrics`. `scripts/evaluate_cf.py --run_dir <cf run> --libs_data_config config/libs_data_measured.yaml` rebuilds the measured token cache with the same line-embedding config, reloads encoder, heads, seeds and layer, and reports spectrum-level, per-sample-median and per-instrument metrics (`evaluation/cf_measured_<timestamp>/`, `test_results_measured` in `run_info.yaml`).

### Optimizer split

AdamW with two parameter groups: encoder at `0.1 × learning_rate`, heads at full `learning_rate` (for `cf_quantification` the two CF heads; the Saha–Boltzmann layer has no parameters and the seed modules are frozen); cosine schedule with linear warmup.

### Checkpointing

`ModelCheckpoint` (`filename='best'`, `save_top_k=1`, `save_last=True`) monitors task-specific metrics: `val/accuracy` (classification, max), `val/reg_mae` (quantification / regression, min), `val/bin_accuracy` (binned, max), `val/det_f1` (detection, max), `val/cf_log_rmse` (cf_quantification, min), or `val/loss` (`both`, min); `--early_stopping` adds `EarlyStopping` on the same metric with patience 10. Each fine-tune run's `checkpoints/` directory receives `best.ckpt` and `last.ckpt` (Lightning checkpoints storing `encoder.*` and head weights, with `pool` in `hyper_parameters`), `encoder_latest.pt` + `encoder_latest_info.txt` (raw encoder state dict rewritten by `SaveRawEncoderCallback` after every validation epoch), and `final_encoder.pt` (raw encoder state dict written at the end of training). Pre-train runs write `best.ckpt` / `last.ckpt` (monitor `val/loss`), `model_latest.pt` + `model_latest_info.txt` (`SaveRawModelCallback`) and `final_model.pt`; `train_finetune.py --pretrain_run_dir` loads `final_model.pt` → `model_latest.pt` → `last.ckpt` in that order. Evaluation reloads the Lightning checkpoint because the heads are needed: `RunManager.get_checkpoint_for_mode('finetune')` (used by `evaluate_model.py` and `analyze_attention_importance.py`) prefers `last.ckpt` and falls back to `final_encoder.pt` / `encoder_latest.pt` only when it is missing, while `make_publication_figures.py`, `publication/inference_runner.py` and the CF seed loader (`finetune_checkpoint_path`) prefer `best.ckpt`.

---

## Training Configurations

### Model / training YAML files

| File | Embedding | GPU | Pretrain epochs | Finetune epochs | Logger |
|------|-----------|-----|-----------------|-----------------|--------|
| `config_libs_smoke.yaml` | intensity | any | 2 | 2 | tensorboard |
| `config_libs_4090.yaml` | intensity | RTX 4090 | 5 | 5 | tensorboard |
| `config_libs_a100.yaml` | intensity | A100 | 60 | 30 | tensorboard |
| `config_libs_token.yaml` | line_token | any | 2 | 2 | tensorboard |
| `config_libs_token_4090.yaml` | line_token | RTX 4090 | 5 | 5 | tensorboard |
| `config_libs_token_linear.yaml` | line_token_linear | any | 2 | 2 | tensorboard |
| `config_libs_token_linear_4090.yaml` | line_token_linear | RTX 4090 | 20 | 20 | tensorboard |
| `config_libs_cf_smoke.yaml` (= `config_libs_token_linear.yaml` + `finetune.cf`) | line_token_linear | any | 2 | 2 | tensorboard |
| `config_libs_cf_4090.yaml` (= `config_libs_token_linear_4090.yaml` + `finetune.cf`) | line_token_linear | RTX 4090 | 20 | 20 | tensorboard |
| `config.yaml` (legacy synthetic generator; `--config` default) | intensity | A100 | 100 | 50 | tensorboard |
| `config_local.yaml` (legacy synthetic generator) | intensity | 16 GB (RTX 4060 Ti) | 10 | 10 | tensorboard |
| `config_a100.yaml` (legacy synthetic generator; d_model 1024, 24 layers) | intensity | A100 | 100 | 50 | wandb |

Pair the `config_libs_*` files with `--libs_data_config config/libs_data.yaml` (or `config/libs_data_smoke.yaml` / `config/libs_data_cf_smoke.yaml` for the smoke configs). Line-token configs also require `--line_embedding_config config/line_embedding.yaml` (or `line_embedding_smoke.yaml`); the CF configs are meant for `config/line_embedding_cf.yaml` (or `line_embedding_cf_smoke.yaml`). The three legacy configs run without `--libs_data_config` and train on the 2048-bin `SyntheticLIBSGenerator` (`config.yaml` is the default `--config` of both training scripts; `config_a100.yaml` is what `run_pretrain.slurm` / `run_finetune.slurm` launch).

The MIP objective is selected by `pretrain.loss` (`mse` by default, or `classification`). `config_libs_token_linear_4090.yaml` and `config_libs_cf_4090.yaml` use `loss: classification`: cross-entropy over discretized targets (`discretization:` block — 256 log-spaced intensity bins, 100 uniform FWHM bins) via `MaskedLineFeatureHead` (`MaskedBinIntensityHead` in bin mode). `config_libs_4090.yaml` pins `loss: mse`; all other configs fall back to `mse`.

The `finetune.cf` block of the CF configs (identical in both) holds the Saha–Boltzmann solver settings forwarded to `cf.layer.SahaBoltzmannLayer` (`n_iter: 3`, `ridge: 1e-6`, `prior_T: 0.1`, `prior_Ne: 0.1`, `sa_correction: true`, `gamma_nm: 0.01`, `eps: 1e-7`, `min_lines: 2`, `reject_sigma: 3.0`, `reject_floor: 0.15`), the loss weights (`lambda_T`, `lambda_Ne`, `lambda_Nl: 0.1`, `lambda_w: 0.01`) and `presence_gate: true`; see [Downstream Fine-Tuning](#downstream-fine-tuning).

### Data pipeline YAML files

| File | Generator | Sample types | Shots / type | Clusters | Binned bins | Split strategy |
|------|-----------|--------------|--------------|----------|-------------|----------------|
| `libs_data.yaml` | physics v2 (`plasma_model: two_zone`, `fine_step_nm: 0.002`, 22 workers) | 2258 (all) | 50 | 10 | 1000 | `random` |
| `libs_data_cf_smoke.yaml` | physics v2 (same knobs, `fine_step_nm: 0.005`, 1 worker) | 3 | 6 | 3 | 50 | `random` |
| `libs_data_smoke.yaml` | legacy v1 (no `plasma_model`; keeps its old cache key) | 3 | 6 | 3 | 50 | `random` (default) |
| `libs_data_measured.yaml` (`source: measured`) | measured Chameleon/OptiCal JSON | 121 measured samples (3,346 JSON files) | 1 spectrum per JSON file (first valid run) | 10 | 1000 | `group_sample` |

`libs_data_measured.yaml` is dispatched by `build_dataset_from_config` to `data/measured_pipeline.py` and cached as `measured_cache_{md5}.h5`; build it with `scripts/build_measured_dataset.py`. The physics-v2 configs are dispatched to `data/two_zone_pipeline.py` (see [Data Sources](#data-sources)); their cache key covers every `generation:` knob plus `physics_version: 2`, so changing any of them regenerates (~2 h for the full config).

### Key Parameters

#### Masking Parameters

```yaml
pretrain:
  mask_ratio: 0.15              # Total fraction masked (used in every embedding mode)
  mask_token_prob: 0.8          # % replaced with learnable mask embedding
  random_token_prob: 0.1        # % replaced with random intensity value
  # Remaining 10% kept unchanged (original intensity)
  # NOTE: mask_token_prob / random_token_prob are not read from the YAML —
  # train_pretrain.py never passes them, so MaskedLIBSDataset always uses its
  # built-in 0.8 / 0.1 defaults.

  # The keys below apply to bin (intensity) mode only; line-token datasets use mask_ratio alone.
  contiguous_masking: true
  block_sizes: [25, 50, 100, 150]   # config_libs_a100.yaml; [25, 50, 100] in config_libs_4090.yaml

  peak_bias_enabled: true
  peak_bias_ratio: 0.5
  peak_threshold: 0.2
```

#### Training Parameters

```yaml
pretrain:                        # values from config_libs_a100.yaml
  loss: mse                      # optional; mse (default) | classification (CE on discretized targets; config_libs_token_linear_4090.yaml, config_libs_cf_4090.yaml)
  batch_size: 16                 # 8 (config_libs_4090), 4 (line-token 4090 and CF configs), 2 (smoke)
  accumulate_grad_batches: 4     # effective batch 64 (a100) / 32 (4090); 8 in the line-token 4090 and CF configs; pre-train Trainer only
  epochs: 60                     # 5 (config_libs_4090, config_libs_token_4090), 20 (config_libs_token_linear_4090, config_libs_cf_4090), 2 (smoke)
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_epochs: 5               # 1 in all 4090 and smoke configs
  min_lr: 1e-6
```

### Dataset Sizes

| Config | Train | Validation | Test |
|--------|-------|------------|------|
| `libs_data_smoke.yaml` / `libs_data_cf_smoke.yaml` (fine-tune, `random`) | 12 | 3 | 3 |
| `libs_data_cf_smoke.yaml` (fine-tune, `group_sample`: one sample type per partition) | 6 | 6 | 6 |
| smoke configs (pre-train, `val_fraction: 0.34`) | 12 | 6 | — |
| `libs_data.yaml` (fine-tune; 2,258 types × 50 shots = 112,900 spectra) | 79,030 | 16,935 | 16,935 |
| `libs_data.yaml` (pre-train, `val_fraction: 0.15`) | 95,965 | 16,935 | — |
| `libs_data_measured.yaml` (fine-tune, `random`; 3,346 spectra) | 2,344 | 501 | 501 |

Fine-tune splits come from `external_data/cache/splits_{md5}.json` (`get_or_make_splits`, fractions from `downstream.splits`; grouped strategies use `splits_{md5}_{strategy}.json` and their sizes follow whole groups, so they only approximate the fractions — `libs_data_measured.yaml` now defaults to `group_sample`). Pre-training does not read that file: `train_pretrain.py` holds out `data.synthetic.val_fraction` of a seeded permutation for validation and trains on the rest (no test set). With the default seed 42 and both fractions at 0.15, the pre-train validation set is exactly the fine-tune `random` test split, and the pre-train training set is fine-tune train + val. The `libs_data.yaml` row assumes every sample type survives generation (the physics-version-2 cache is keyed differently from the legacy one, but has the same shot count).

### SLURM deployment

| Script | Purpose |
|--------|---------|
| `run_pretrain_libs.slurm` | Bin-embedding pretrain (`config_libs_a100.yaml` + `libs_data.yaml`) |
| `run_finetune_libs.slurm` | Binned finetune (`quantification_binned`, `cls_mean`) |
| `run_eval.slurm` | Post-training evaluation (`evaluate_model.py`, legacy generator; see [Evaluation Metrics](#evaluation-metrics)) |
| `run_pretrain.slurm` | Legacy synthetic-generator pretrain (`config_a100.yaml`, no `--libs_data_config`) |
| `run_finetune.slurm` | Legacy synthetic-generator finetune (`config_a100.yaml`, `--task both`, no `--libs_data_config`) |

For line-token runs, mirror these scripts with `config_libs_token_4090.yaml` (runtime embedding) or `config_libs_token_linear_4090.yaml` (pre-baked tokens) and `--line_embedding_config config/line_embedding.yaml`. The end-to-end CF pipeline (data regeneration, CF tokens, pretrain, seeds, CF fine-tune, measured evaluation) is scripted for a single workstation in `scripts/run_cf_pipeline.sh`.

All SLURM jobs run in the `sslibs` enroot container with the workspace mounted at `/workspace`.

---

## Evaluation Metrics

### Pre-training (MIP)

| Metric | Description |
|--------|-------------|
| MSE | Mean squared error on masked bins (`evaluate_model.py`; also `val/loss` when `pretrain.loss: mse`) |
| RMSE | Root MSE (`evaluate_model.py` only) |
| MAE | Mean absolute error on masked positions (`val/mae`); with `pretrain.loss: classification` it is computed on values decoded from the argmax bin |
| R² | Coefficient of determination on masked positions (`val/r2`) |
| `val/bin_accuracy` | Classification-loss MIP only: fraction of masked positions whose argmax bin equals the discretized target (mean of intensity and FWHM channels for line tokens) |
| `val/mae_intensity`, `val/mae_fwhm` | Classification-loss line-token MIP only: per-channel decoded MAE |
| Peak Coverage | Masking statistic (not a model metric): fraction of masked bins that lie on peaks, printed once by `PretrainDataModule.setup()` in bin mode |

### Fine-tuning (Classification)

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy (`{stage}/accuracy`) |
| Balanced Accuracy | Macro-averaged per-class accuracy (`{stage}/balanced_accuracy`) |
| F1 (macro) | Macro F1 score (`evaluate_model.py` only) |
| F1 (weighted) | Support-weighted F1 score (`evaluate_model.py` only) |
| Precision / Recall (macro) | Macro-averaged precision and recall (`evaluate_model.py` only) |
| Per-class accuracy | `accuracy_<class>` for every class present in the test set (`evaluate_model.py` only) |
| Confusion Matrix | Per-class prediction analysis (`evaluate_model.py` only) |

### Fine-tuning (Regression / quantification)

| Metric | Description |
|--------|-------------|
| MSE | Mean squared error (`{stage}/reg_loss`) |
| RMSE | Root MSE (`evaluate_model.py` only) |
| MAE | Mean absolute error (`{stage}/reg_mae`) |
| R² | Per-element R² scores (averaged) (`{stage}/reg_r2`) |
| Per-element MSE / MAE / R² | `mse_<name>`, `mae_<name>`, `r2_<name>` for each output (`evaluate_model.py` only) |

### Fine-tuning (Binned quantification)

| Metric | Description |
|--------|-------------|
| `bin_loss` | Per-element cross-entropy over concentration bins (the training loss) |
| `bin_accuracy` | Fraction of (sample, element) pairs with correct argmax bin |
| `decoded_mae` | MAE after mapping predicted bins back to [0, 1] concentrations |
| `decoded_r2` | Mean of per-element R² on decoded concentrations — primary signal for whether the model beats the per-element mean |
| `test/per_element/<El>/{mae,r2,pearson,spearman}` | Test-only per-element diagnostics on decoded concentrations (plus a `target_hist` histogram); also saved with `n_samples` under `test_results.per_element` in `run_info.yaml` |

`bin_accuracy` alone can be misleading when most elements are near-zero (predicting bin 0 scores well). Prefer `decoded_mae` and `decoded_r2` for model selection.

### Fine-tuning (Element detection)

Presence targets are derived on the fly as `concentration >= LOD` per element (`concentration_to_presence`; LODs from `--element_lod_config`, default `config/element_lod.yaml`). Predictions are `sigmoid(logit) >= 0.5`.

| Metric | Description |
|--------|-------------|
| `det_loss` | Multi-label BCE-with-logits (optional per-element `pos_weight`) |
| `det_accuracy` | Fraction of correct (sample, element) presence decisions |
| `det_precision` / `det_recall` / `det_f1` | Micro-averaged over all element decisions; `val/det_f1` is the checkpoint monitor |
| `det_exact_match` | Fraction of spectra with every element decided correctly |
| `test/detection/{macro_f1,micro_f1,element_accuracy,exact_match}` | Test-only aggregates; `run_info.yaml` `test_results.detection` also stores macro/micro precision and recall |
| `test/detection/per_element/<El>/f1` | Test-only per-element F1; `test_results.detection.per_element` also stores `precision`, `recall`, `accuracy`, `support`, `lod`, `n_samples` |

### Fine-tuning (Calibration-free quantification)

All concentration metrics are computed in log space on `ln(C + eps)` (`finetune.cf.eps`, default 1e-7) and only over (spectrum, element) pairs whose **true** concentration is at or above the per-element LOD ("uncensored truth"); the loss additionally penalises predictions above the LOD for censored pairs (see [Downstream Fine-Tuning](#downstream-fine-tuning)). Concentration metrics are logged for every batch that carries concentrations (synthetic or measured); the plasma metrics only for synthetic shots (`has_plasma_labels == 1`).

| Metric | Description |
|--------|-------------|
| `cf_loss` | Total CF training loss (zero, but differentiable, on batches without plasma labels) |
| `cf_log_rmse` | RMSE of `ln C_pred − ln C_true` over uncensored pairs; `val/cf_log_rmse` (min) is the checkpoint monitor |
| `cf_within2x` | Fraction of uncensored pairs predicted within a factor of 2 of the truth |
| `cf_r2_major` | Mean per-element R² (linear space) over the major elements present in `element_names` (`CF_MAJOR_ELEMENTS` = Fe, C, Mn, Si, Cr, Ni, Cu, Al), per batch |
| `cf_mean_weight` | Mean learned line weight over fit-valid lines |
| `cf_n_censored` | Mean number of elements per spectrum the solver reports as censored (below LOD) |
| `cf_conc_loss`, `cf_T_loss`, `cf_Ne_loss`, `cf_Nl_loss`, `cf_w_loss` | The individual loss terms |
| `te_mape`, `ne_log_mae` | Relative error of the solved T and absolute error of the solved log10 Ne against the generator's inner-zone values, one-zone synthetic shots only (two-zone T is ill-defined) |
| `nl_log_mae` | Absolute error of the plasma-init head's `log10 N·l0` against the generator's `log10(N1·l_inner)`, all synthetic shots |
| `test/per_element/<El>/{mae,r2,pearson,spearman,log_rmse,within_2x,n_censored}` | Test-only per-element diagnostics (`run_info.yaml` `test_results.per_element` adds `n_samples`, `n_uncensored_truth`, `lod`) |
| `test/plasma/<key>` | Test-only plasma recovery over labelled test shots: `n_test`, `n_labelled`, `n_one_zone`, `n_two_zone`, `te_mape`, `te_rmse`, `ne_log_mae`, `ne_log_rmse`, `t0_mape`, `ne0_log_mae` (init-head errors), `nl_log_mae`, `te_mape_two_zone`; saved as `test_results.test_plasma_metrics` |

Measured spectra never enter CF training; they are scored zero-shot with `scripts/evaluate_cf.py`, which writes `per_spectrum.csv`, `per_element.yaml` (spectrum-level **and per-sample-median** `mae`, `log_rmse`, `within_2x`, `r2`, `n`, `n_censored` per element, CF solver and binned seed side by side, plus macro / major-element summaries and median plasma state) and `per_instrument.yaml` (the same CF metrics per instrument, grouped with `measured_groups(by='instrument')`) under `<run_dir>/evaluation/cf_measured_<timestamp>/`, and appends `test_results_measured` to the run's `run_info.yaml`.

`evaluate_model.py` (the script `run_eval.slurm` runs; `--mode pretrain|finetune`, auto-detected from the run directory) is a legacy-path tool: it scores 1,000 freshly generated `SyntheticLIBSGenerator` spectra (`generate_test_data`, seed 9999; the run's `config.yaml` must contain `data.synthetic.noise_sigma`, `peak_width_range` and `intensity_variation`), not the cached LIBS test split. Its fine-tune mode loads only `classification_head` / `regression_head` from `last.ckpt` and raises `NotImplementedError` for encoders whose `embedding_type` is not `intensity`. Judge `quantification_binned`, `detection` and `cf_quantification` runs from TensorBoard logs and `run_info.yaml` `test_results` (aggregate `test/*` metrics plus the `per_element` / `detection` / `test_plasma_metrics` blocks), or run checkpoint inference on the real test split with `make_publication_figures.py` (loads the full `LIBSFinetuneModule` including `binned_head` / `detection_head`; `fig4_pred_vs_true`, `fig4b_per_element_r2`), `make_comparison_figures.py` (raw `intensity` vs tokenized `line_token_linear` detection runs, via `publication/inference_runner.py`), `analyze_attention_importance.py`, and `scripts/evaluate_cf.py` for CF runs.

---

## How BERT-style Masking Works for Continuous Spectra

In standard BERT (discrete tokens), 15% of tokens are selected for prediction and
the input is modified using the 80/10/10 rule. For continuous spectral intensities,
we implement this at two levels:

### Step 1: Dataset-level input modification (`MaskedLIBSDataset`)

The dataset selects 15% of bins (using contiguous + peak-biased strategy) and
modifies the input spectrum:

- **80% (type 1):** Intensity set to 0 (placeholder — will be replaced by mask embedding)
- **10% (type 2):** Intensity replaced with a random value in [0, 1]
- **10% (type 3):** Intensity kept unchanged

### Step 2: Embedding-level mask token insertion (`SpectralEmbedding`)

The embedding layer projects all intensity values to `d_model` vectors, then
replaces **only the type-1 positions** (the 80%) with a learnable mask token
embedding. Type-2 and type-3 positions keep their projected intensity embeddings.

```
Type 1 (80%): intensity=0 → project → REPLACED with learnable [MASK] embedding
Type 2 (10%): intensity=random → project → keeps projected random embedding
Type 3 (10%): intensity=original → project → keeps projected original embedding
```

### Step 3: Loss computation

With the default `pretrain.loss: mse`, MSE loss is computed on **all 15%** of masked
positions (types 1, 2, and 3), comparing the MIP head predictions against the
original (unmasked) intensities.

With `pretrain.loss: classification`, the same masked positions are scored with
cross-entropy instead: the scalar MIP head is replaced by `MaskedBinIntensityHead`
(one logit per intensity class, `model.num_intensity_bins`, default 256; the model
emits `intensity_logits` instead of `mip_predictions`), and the original intensities
are discretized on the fly by `SpectroscopicDiscretizer` (`data/discretization.py`,
configured by the optional `discretization:` block — the built-in fallback is 256
log-spaced intensity bins over 1e-4..1.0). Line-token modes use `MaskedLineFeatureHead`,
which adds `fwhm_logits` (100 uniform FWHM bins) and sums the two CE terms. Validation
additionally logs `val/bin_accuracy`. Of the shipped configs only
`config_libs_token_linear_4090.yaml` and `config_libs_cf_4090.yaml` enable it;
`config_libs_4090.yaml` carries the `discretization:` block but keeps `loss: mse`.

### Why does this matter?

The 10% random + 10% unchanged strategy prevents the model from learning a
shortcut like "predict only when I see [MASK]." Since the model must also
correctly predict at positions where it sees random or original values, it is
forced to learn genuine spectral context — understanding that certain emission
lines co-occur, that peak ratios imply specific compositions, etc. This
produces representations that transfer well to downstream tasks where no masking
is applied.

---

## Design Decisions & Rationale

### Why Contiguous + Peak-Biased Masking?

1. **Sparse spectra problem**: ~95% of bins are noise. Random masking tests "predict noise" which is trivial.

2. **Local interpolation problem**: If individual bins are masked, the model can just average neighbors. Contiguous blocks force global reasoning.

3. **Variable sizes prevent overfitting**: Fixed 50-bin blocks could lead to pattern memorization. Variable sizes (25-150) create diverse training.

### Why Not Higher Mask Ratios?

We use 15% (BERT default), but for sparse data, higher ratios (30-40%) may work better. This is configurable:

```yaml
mask_ratio: 0.30  # 30% masking
```

### Why Transformer over CNN?

1. **Global context**: Attention can relate distant wavelengths (element correlations)
2. **Position flexibility**: Positional encoding handles wavelength information
3. **Transfer learning**: Transformer embeddings transfer well to downstream tasks

---

## Usage Examples

### Running pre-training

```bash
# config/libs_data.yaml is the physics-v2 generator: the first run of any of these commands
# materialises synthetic_cache_<key>.h5 (~2 h with 22 workers); later runs hit the cache.

# Bin embedding — RTX 4090
uv run python train_pretrain.py \
    --config config/config_libs_4090.yaml \
    --libs_data_config config/libs_data.yaml \
    --experiment_name libs_bin_pretrain \
    --num_workers 4

# Line-token embedding — RTX 4090 (builds line caches on first run)
uv run python train_pretrain.py \
    --config config/config_libs_token_4090.yaml \
    --libs_data_config config/libs_data.yaml \
    --line_embedding_config config/line_embedding.yaml \
    --experiment_name libs_line_pretrain \
    --num_workers 4

# Line-token-linear — pre-baked tokens (optional standalone build first)
uv run python scripts/build_line_tokens.py \
    --libs_data_config config/libs_data.yaml \
    --line_embedding_config config/line_embedding.yaml
uv run python train_pretrain.py \
    --config config/config_libs_token_linear_4090.yaml \
    --libs_data_config config/libs_data.yaml \
    --line_embedding_config config/line_embedding.yaml \
    --experiment_name libs_token_linear_pretrain \
    --num_workers 4

# A100 bin embedding via SLURM
sbatch run_pretrain_libs.slurm
```

### Running fine-tuning

```bash
# Binned quantification — bin embedding (match pretrain config)
uv run python train_finetune.py \
    --config config/config_libs_4090.yaml \
    --pretrain_run_dir runs/pretrain_<timestamp>_libs_bin_pretrain \
    --libs_data_config config/libs_data.yaml \
    --task quantification_binned \
    --pool cls_mean \
    --num_workers 4

# Binned quantification — line-token (match pretrain + line_embedding_config)
uv run python train_finetune.py \
    --config config/config_libs_token_4090.yaml \
    --pretrain_run_dir runs/pretrain_<timestamp>_libs_line_pretrain \
    --libs_data_config config/libs_data.yaml \
    --line_embedding_config config/line_embedding.yaml \
    --task quantification_binned \
    --pool cls_mean

sbatch run_finetune_libs.slurm runs/pretrain_<timestamp>_libs_pretrain_a100
```

### Calibration-free quantification (physics-v2 data, CF dictionary)

```bash
# 0. sanity checks (no training): generator physics, CF solver / torch layer
uv run python scripts/check_two_zone_physics.py --libs_data_config config/libs_data_cf_smoke.yaml
uv run python scripts/check_cf_layer.py

# 1. tokens: physics-v2 spectra (generated on first use) + cf_isolated dictionary + Voigt fits
uv run python scripts/build_line_tokens.py --libs_data_config config/libs_data.yaml \
    --line_embedding_config config/line_embedding_cf.yaml
uv run python scripts/build_line_tokens.py --libs_data_config config/libs_data_measured.yaml \
    --line_embedding_config config/line_embedding_cf.yaml        # measured spectra, same dictionary

# 2. pretrain the line_token_linear encoder on the CF tokens
uv run python train_pretrain.py --config config/config_libs_token_linear_4090.yaml \
    --libs_data_config config/libs_data.yaml --line_embedding_config config/line_embedding_cf.yaml \
    --experiment_name cf_pretrain --num_workers 4

# 3. seeds on the SAME token cache: binned quantification (C0) and detection (presence gate)
uv run python train_finetune.py --config config/config_libs_token_linear_4090.yaml \
    --pretrain_run_dir runs/pretrain_<ts>_cf_pretrain --libs_data_config config/libs_data.yaml \
    --line_embedding_config config/line_embedding_cf.yaml --task quantification_binned --pool cls_mean \
    --experiment_name cf_seed_binned
uv run python train_finetune.py ... --task detection --experiment_name cf_seed_detection

# 4. CF task (learned weights) and the zero-parameter reference (--cf_pure_physics)
uv run python train_finetune.py --config config/config_libs_cf_4090.yaml \
    --pretrain_run_dir runs/pretrain_<ts>_cf_pretrain --libs_data_config config/libs_data.yaml \
    --line_embedding_config config/line_embedding_cf.yaml --task cf_quantification --pool cls_mean \
    --seed_binned_run_dir runs/finetune_<ts>_cf_seed_binned \
    --seed_detection_run_dir runs/finetune_<ts>_cf_seed_detection \
    --experiment_name cf_learned --num_workers 4
uv run python train_finetune.py ... --task cf_quantification --cf_pure_physics --experiment_name cf_pure

# 5. zero-shot evaluation on measured spectra (never used for training)
uv run python scripts/evaluate_cf.py --run_dir runs/finetune_<ts>_cf_learned \
    --libs_data_config config/libs_data_measured.yaml --line_embedding_config config/line_embedding_cf.yaml
uv run python scripts/run_cf_classical.py --tokens external_data/cache/line_tokens_<hash>.h5 \
    --spectra_cache external_data/cache/measured_cache_<hash>.h5 \
    --line_dict external_data/cache/line_dict_<hash>.h5 --line_list cf_oes54   # classical CF, 54 curated lines
```

Smoke-sized equivalents use `config/config_libs_cf_smoke.yaml`, `config/libs_data_cf_smoke.yaml` and `config/line_embedding_cf_smoke.yaml`; `scripts/run_cf_pipeline.sh` chains steps 1–5 (plus attention-importance and publication figures) on one workstation.

### Evaluation and run discovery

```bash
uv run python list_runs.py --latest
uv run python evaluate_model.py --run_dir runs/pretrain_<timestamp>_...
uv run python evaluate_model.py --run_dir runs/finetune_<timestamp>_... --mode finetune   # legacy generator, intensity encoders only
```

Runs are stored under `runs/{type}_{timestamp}_{experiment}/` with `config.yaml`, `run_info.yaml`, `checkpoints/`, `logs/`, and optional `evaluation/`. See `PROJECT_SUMMARY.json` / `PROJECT_SUMMARY.html` for full layout and CLI reference.

### Checking masking statistics (bin mode)

Load cached spectra from the libs pipeline, wrap in `MaskedLIBSDataset`, and call `get_masking_stats(n_samples=100)` — see `PretrainDataModule.setup()` logging during training.

---

## Future Improvements

1. **Curriculum masking**: Start with low mask ratio, increase over training
2. **Adaptive peak detection**: Learn peak threshold from data
3. **Contrastive pre-training**: Add SimCLR-style objective
4. **Multi-scale masking**: Mask at different resolutions

---

## References

- BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)
- SpanBERT: Improving Pre-training by Representing and Predicting Spans (Joshi et al., 2020)
- DreaMS: Deep Representations for Mass Spectrometry (inspiration for spectral transformers)
