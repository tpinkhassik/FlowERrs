# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FlowER** is a deep learning model for electron flow matching in chemical reaction mechanism prediction, published in *Nature* (2025). It predicts mechanistic pathways by modeling electron redistribution using flow matching, conceptually aligned with arrow-pushing formalisms in organic chemistry.

## Environment Setup

```bash
conda create -n flower python=3.10
conda activate flower
pip install -r requirements.txt
```

GPU with at least 25GB VRAM and CUDA >= 12.2 is required.

## Key Commands

The workflow is orchestrated through `run_FlowER_large_(old|new)Data.sh`. Use comments to switch between modes:

```bash
# Train
sh run_FlowER_large_newData.sh   # with scripts/train.sh uncommented

# Validate (set do_validate=True in settings.py)
sh run_FlowER_large_newData.sh   # with scripts/eval_multiGPU.sh uncommented

# Test (set do_validate=False in settings.py, set MODEL_NAME)
sh run_FlowER_large_newData.sh   # with scripts/eval_multiGPU.sh uncommented

# Beam search
sh run_FlowER_large_newData.sh   # with scripts/search_multiGPU.sh uncommented
```

Data and checkpoints are downloaded from [Figshare](https://figshare.com/articles/dataset/FlowER_-_Mechanistic_datasets_and_model_checkpoint/28359407/3) and placed under `FlowER/data/` and `FlowER/checkpoints/` respectively.

## Architecture

### Two Central Files

The entire workflow revolves around two files:
1. **`run_FlowER_large_(old|new)Data.sh`** — sets environment variables (data paths, GPU config, experiment name) and selects which script to run by commenting/uncommenting
2. **`settings.py`** — the `Args` class centralizes all hyperparameters; switching modes (training vs. validation vs. inference vs. beam search) is done by commenting/uncommenting sections within this file

### Model (`model/attn_encoder.py`)

`AttnEncoderXL` is a transformer-based graph encoder:
- Atom embeddings (periodic table) + RBF bond distance expansion + sinusoidal time embeddings
- 12-layer multi-headed relative attention transformer
- Two output heads: bond-electron (BE) matrix prediction + chirality vector (CV) prediction

### Flow Matching (`model/flow_matching.py`)

`ConditionalFlowMatcher` implements conditional flow matching from optimal transport literature. It samples from `P(x_t | x0, x1) = N(t*x1 + (1-t)*x0, σ²I)` and computes vector fields. Zero-centering enforces chemical validity (electron conservation).

### Training (`train.py`)

Uses PyTorch DDP for multi-GPU training. The loop:
1. Converts SMILES → atom-mapped molecules → BE matrices + chiral vectors
2. Samples from the conditional probability path at time `t`
3. Encodes source with time embedding via `AttnEncoderXL`
4. Computes MSE loss between predicted and target vector fields (BE + CV branches)
5. Saves checkpoints every `save_iter` steps with NoamLR scheduling

### Inference (`eval_multiGPU.py`)

Generates multiple samples per input using an ODE solver (torchdiffeq), rounds predicted matrices using electron-conserving rounding (`utils/rounding.py`), then reconstructs molecules from BE matrices.

### Beam Search (`beam_predict_multiGPU.py`)

Explores mechanistic pathways using beam search on a networkx graph. Expands top-`beam_size` candidates by cumulative probability up to `max_depth` steps, returning `nbest` final pathways. Visualize results in `examples/vis_network.ipynb`.

### Data (`utils/data_utils.py`)

`ReactionDataset` converts atom-mapped SMILES to BE matrices. Bond electron counts: single=2, double=4, triple=6, aromatic=3. Input format for training/eval: `mapped_reaction|sequence_idx`. Input format for beam search: `reactants>>product1|product2|...`.

## Key Configuration (`settings.py`)

Important settings to be aware of when switching modes:

| Setting | Purpose |
|---|---|
| `do_validate` | `True` = validation set, `False` = test set |
| `steps2validate` | List of checkpoint step numbers to evaluate |
| `MODEL_NAME` | Checkpoint path for testing |
| `sample_size` | Samples generated per input (scale via `SCALE` env var) |
| `beam_size`, `nbest`, `max_depth` | Beam search parameters |
| `sigma` | Gaussian noise std for BE matrix reparameterization (default 0.15) |

## Data Paths

In `run_FlowER_large_newData.sh`, data paths default to `/home/ptim/orcd/scratch/` (HPC cluster). Override with environment variables before running:
```bash
export TRAIN_FILE=/path/to/train.txt
export VAL_FILE=/path/to/val.txt
export TEST_FILE=/path/to/test.txt
```
