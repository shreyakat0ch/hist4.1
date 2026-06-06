# Configuration (`config.py`)

All hyperparameters, paths, and model dimensions are centralized in `config.py`. Modifying this file alters the behavior of the entire pipeline.

## Core Dimensions

```python
WINDOW_SIZE = 32768      # Input DNA length in base pairs
RNA_INPUT_DIM = 4000     # Number of top variable genes
NUM_MARKS = 4            # Number of target histone marks
```

## Architecture Hyperparameters

```python
D_MODEL = 256            # Hidden dimension of CNN/Mamba
N_LAYERS = 4             # Number of BiMamba layers
D_STATE = 32             # Mamba SSM state dimension
MAMBA_EXPAND = 2         # Mamba internal expansion factor

RNA_HIDDEN_DIM = 512     # Output dimension of RNA MLP
NUM_RNA_TOKENS = 8       # Number of pseudo-tokens for Cross-Attention
CROSS_ATTN_HEADS = 4     # Number of attention heads
HEAD_HIDDEN_DIM = 256    # Hidden dim of final regression MLPs

DROPOUT = 0.3            # Dropout rate throughout network
```

## Track Head Parameters

```python
TRACK_BIN_SIZE = 64      # Resolution in bp (must match CNN stride)
NUM_TRACK_BINS = 512     # 32768 / 64 = 512 output bins
TRACK_LOSS_WEIGHT = 0.3  # Auxiliary loss weight
```

## Data Sampling & Augmentation

```python
PEAK_FRAC = 0.50         # Ratio of peak-centered vs background windows
BG_MARGIN = 20000        # Distance from any peak for background (bp)
MAX_PEAKS_PER_MARK_PER_CELL = 3000  # Cap (or target for oversampling)

AUGMENT_RC = True        # 50% chance of reverse complement
AUGMENT_OFFSET_MAX_BP = 2000  # Translation jitter (± bp)
```

## Training Hyperparameters

```python
BATCH_SIZE = 256         # Physical batch size (memory constrained)
GRAD_ACCUM_STEPS = 4     # Effective batch = 256 * 4 = 1024
LEARNING_RATE = 3e-4     # Max learning rate
WEIGHT_DECAY = 5e-2      # L2 regularization
WARMUP_STEPS = 3000      # Steps for linear LR warmup
EPOCHS = 50              # Maximum training epochs
EARLY_STOPPING_PATIENCE = 10  # Epochs without improvement to halt
```

## Mark-Specific Contexts

Different marks pool features from different window sizes:

```python
MARK_CONTEXT_BP = {
    "H3K27ac":  16384,   # Enhancers: medium-broad
    "H3K4me3":   4096,   # Promoters: sharp/narrow
    "H3K27me3": 32768,   # Polycomb: ultra-broad domains
    "H3K9me3":  16384,   # Heterochromatin: medium-broad
}
```

## Ablation Testing

```python
# Set to True to zero-out the RNA vector, effectively turning Deep-H 
# into a DNA-only model. Used to prove the value of RNA conditioning.
RNA_ABLATION = False
```

---

**Next:** [Inference →](../inference.md)
