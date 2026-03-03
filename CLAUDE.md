# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlowER (Flow Matching for Electron Redistribution) models chemical reactions as electron redistribution using flow matching, aligned with arrow-pushing formalisms. Published in *Nature* 645, 115–123 (2025). DOI: 10.1038/s41586-025-09426-9

## Environment Setup

```bash
conda create -n flower python=3.10
conda activate flower
pip install -r requirements.txt        # CUDA >= 12.2
pip install -r requirements_rocm.txt   # AMD ROCm alternative
```

Requires GPU with >= 25GB VRAM.

## Key Commands

All workflows are driven by orchestration scripts that set environment variables, then call shell wrappers around `torchrun`. The two orchestration scripts are `run_FlowER_large_newData.sh` and `run_FlowER_large_oldData.sh`.

**Training:**
```bash
sh run_FlowER_large_newData.sh   # with scripts/train.sh uncommented
```

**Validation/Testing:**
```bash
sh run_FlowER_large_newData.sh   # with scripts/eval_multiGPU.sh uncommented
```

**Beam Search (mechanism prediction):**
```bash
sh run_FlowER_large_newData.sh   # with scripts/search_multiGPU.sh uncommented
```

Workflow switching is done by commenting/uncommenting lines at the bottom of the orchestration script and toggling `do_validate` in `settings.py`.

There is no test suite, linter, or build system — this is a research codebase.

## Configuration

**`settings.py`** — Central config via the `Args` class. Most values come from environment variables set in the orchestration shell scripts. Key settings:
- `do_validate = True` → validation mode (iterates `steps2validate` checkpoints)
- `do_validate = False` → test/inference mode (uses `MODEL_NAME` checkpoint)
- `SCALE` env var controls `sample_size = 64 // SCALE` (use 4 for training validation, 1 for final testing)
- Beam search params: `beam_size`, `nbest`, `max_depth`, `chunk_size`

**`run_FlowER_large_newData.sh`** — Sets all env vars (`DATA_NAME`, `EXP_NAME`, `EMB_DIM`, `SIGMA`, GPU config, file paths). Default paths point to MIT ORCD scratch storage (`/home/ptim/orcd/scratch/`).

## Architecture

### Data Representation
- Reactions represented as **Bond-Electron (BE) matrices**: symmetric matrices where entries are electron counts between atom pairs (single=2, double=4, triple=6, aromatic=3)
- Input format for train/val/test: `mapped_reaction|sequence_idx` (atom-mapped SMILES)
- Input format for beam search: `reactants>>product1|product2|...` (unmapped SMILES)
- Padding value: `MATRIX_PAD = -30` (defined in `utils/data_utils.py`)

### Core Modules

**`model/attn_encoder.py` — `AttnEncoderXL`**: Transformer model with RBF expansion of the BE matrix, sinusoidal timestep embeddings, relative positional embeddings. Two output heads: BE matrix velocity field and chiral vector velocity field.

**`model/flow_matching.py` — `ConditionalFlowMatcher`**: Implements conditional flow matching. Samples intermediate states along probability path `P(x_t|x0,x1) = N(t*x1 + (1-t)*x0, σ²I)`. Zero-centering enforces electron conservation — this is a critical chemical constraint.

**`utils/data_utils.py` — `ReactionDataset`**: Converts atom-mapped SMILES → BE matrices + chiral vectors. Handles batching with sorting/bucketing and padding masks. `BEmatrix_to_mol()` reconstructs molecules from predicted BE matrices.

**`utils/rounding.py` — `saferound_tensor()`**: Electron-conserving rounding of predicted BE matrices. Preserves total electron count while producing integer bond orders. Critical for chemical validity.

**`utils/stereo_utils.py`**: Chirality handling — atom stereochemistry matching between reactants/products, chiral vector representation.

### Pipeline Flow

1. **Training (`train.py`)**: DDP multi-GPU. Samples noise → predicts velocity fields → MSE loss on BE + chiral branches → AdamW + NoamLR. Checkpoints every `save_iter` steps.
2. **Inference (`eval_multiGPU.py`)**: Loads checkpoint → repeats input `sample_size` times → ODE integration via `torchdiffeq.odeint()` from t=0→1 → electron-conserving rounding → molecule reconstruction → top-k accuracy.
3. **Beam Search (`beam_predict_multiGPU.py`)**: `expand()` generates trajectories → `select()` ranks by cumulative probability → builds networkx DiGraph of reaction pathways → returns top-`nbest` complete sequences.
4. **Sequence Evaluation (`sequence_evaluation.py`)**: Post-processes multi-step predictions grouped by `sequence_idx`.

### Alternative Model Variants
`model/attn_encoder_chiral_aux.py` and `model/attn_encoder_chiral_flow.py` are alternative encoder architectures for chirality handling. The main model is `attn_encoder.py`.

## Data Layout

```
data/{DATASET_NAME}/{train,val,test,beam}.txt
checkpoints/{DATASET_NAME}/{EXP_NAME}/model.{STEP}_{IDX}.pt
```

Data and checkpoints are gitignored. Download from [Figshare](https://figshare.com/articles/dataset/FlowER_-_Mechanistic_datasets_and_model_checkpoint/28359407/3).

## Key Dependencies

- `torch` (2.4.0 CUDA / 2.10.0 ROCm) — model + DDP
- `torchdiffeq` — ODE integration during inference
- `rdkit` — molecular graph construction and SMILES handling
- `networkx` — beam search graph
- `iteround` — rounding utilities
