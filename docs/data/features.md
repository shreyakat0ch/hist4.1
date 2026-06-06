# Feature Engineering

Once windows are sampled, Deep-H extracts features and applies augmentations to create model-ready tensors. This page explains each feature type and the augmentation strategy.

## Per-Window Features

For each sampled window `(cell_line, chromosome, center)`, Deep-H extracts:

### 🧬 DNA Input: One-Hot Encoding

The 32,768 bp DNA sequence is converted into a `[4, 32768]` one-hot tensor:

```python
# Nucleotide encoding: A=0, C=1, G=2, T=3, N=4
# One-hot: row 4 (N) maps to all-zeros
_ONE_HOT_TABLE = np.eye(5, 4, dtype=np.float32)

# Example:  A  C  G  T  N
#          [1, 0, 0, 0]
#          [0, 1, 0, 0]
#          [0, 0, 1, 0]
#          [0, 0, 0, 1]
#          [0, 0, 0, 0]  ← N (unknown) = zero vector
```

!!! tip "GPU-Optimized Encoding"
    Deep-H stores DNA as **uint8 indices** and performs one-hot encoding lazily on the GPU via `F.one_hot()`. This reduces CPU→GPU transfer bandwidth by 4× compared to sending pre-encoded float tensors.

    ```python
    # In model.py forward():
    if dna_seq.dtype == torch.uint8 or dna_seq.dtype == torch.long:
        dna_seq = F.one_hot(dna_seq.long(), num_classes=5
                 )[:, :, :4].permute(0, 2, 1).float()
    ```

### 📊 RNA Input: Expression Vector

The cell line's RNA fingerprint is a pre-computed `[4000]` float vector:

```python
rna = self.rna_map.get(cell, self._dummy_rna)  # [4000]
```

| Property | Value |
|----------|-------|
| Dimension | 4,000 (top variable genes) |
| Scale | log₂(TPM + 1) |
| Missing cells | Zero vector (rare) |
| Normalization | Pre-normalized across dataset |

### 🎯 Scalar Target: Peak Intensity

For each of the 4 histone marks, the target is the **log₂(signal + 1)** value at the window center:

```python
raw_target = self._cell_index[cell].lookup_signal(
    chrom, start, start + config.WINDOW_SIZE
)  # [4] - one value per mark
```

!!! note "Why log₂(signal + 1)?"
    ChIP-seq signal values span 4+ orders of magnitude. Log-transformation compresses this range and makes Huber loss equally sensitive to fold-changes at all intensity levels. The `+1` prevents log(0).

### 📍 Track Target: Binary Peak Map

A `[4, 512]` binary tensor indicating which of the 512 spatial bins (64 bp each) overlap a peak:

```python
track_target, track_valid = self._cell_index[cell].lookup_track(
    chrom, start, start + config.WINDOW_SIZE,
    bin_size=config.TRACK_BIN_SIZE  # 64 bp
)
# track_target: [4, 512] - binary (0/1)
# track_valid:  [4]      - which marks have BED data
```

```
Window: |-------- 32,768 bp --------|
Bins:   |1|2|3|4|...............512|   (64 bp each)
Track:  |0|0|1|1|1|1|0|0|0|1|1|0|...|  ← 1 = peak present
```

### 🎭 Validity Mask

A `[4]` binary mask indicating which marks have ground-truth data for this cell line:

```python
mask = (raw_target != -1.0).astype(np.float32)  # [4]
# Example: [1, 1, 0, 1] means H3K27me3 data is missing
```

---

## Data Augmentation

Deep-H applies two augmentations to improve generalization:

### 1. Random Offset (±2,000 bp)

```python
if self.training and config.AUGMENT_OFFSET_MAX_BP > 0:
    center += random.randint(
        -config.AUGMENT_OFFSET_MAX_BP,  # -2000
         config.AUGMENT_OFFSET_MAX_BP   # +2000
    )
```

!!! example "Why Random Offset?"
    The model should learn that a peak is a peak regardless of its exact position within the 32 kb window. A ±2 kb jitter (6% of window size) forces the model to be translation-invariant without losing the peak from the window entirely.

### 2. Reverse Complement (50% Probability)

```python
if self.training and config.AUGMENT_RC and random.random() < 0.5:
    # Complement: A↔T, C↔G
    comp_table = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
    dna_indices = comp_table[dna_indices[::-1]].copy()
    # Flip track target to match
    track_target = track_target[:, ::-1].copy()
```

!!! info "Biological Rationale"
    DNA is double-stranded. A histone mark at a genomic position exists on **both strands**. By training on both orientations, the model learns strand-invariant features, effectively doubling the training data.

---

## Final Tensor Shapes

The DataLoader yields batches of:

| Tensor | Shape | Type | Description |
|--------|-------|------|-------------|
| `dna` | `[B, 32768]` | uint8 | DNA indices (one-hot on GPU) |
| `rna` | `[B, 4000]` | float32 | RNA expression vector |
| `target` | `[B, 4]` | float32 | Scalar targets (log₂ scale) |
| `mask` | `[B, 4]` | float32 | Mark availability mask |
| `track_target` | `[B, 4, 512]` | float32 | Binary track labels |
| `track_valid` | `[B, 4]` | float32 | Track availability mask |

```python
# DataLoader configuration
train_loader = DataLoader(
    train_dataset,
    batch_size=256,         # 256 windows per micro-batch
    shuffle=True,
    num_workers=16,         # 16 parallel data loading workers
    pin_memory=True,        # Pre-stage tensors in pinned CPU memory
    persistent_workers=True # Keep workers alive between epochs
)
```

<div class="stats-row">
  <div class="stat-card stat-coral">
    <span class="stat-value">256</span>
    <span class="stat-label">Batch Size</span>
  </div>
  <div class="stat-card stat-turquoise">
    <span class="stat-value">×4</span>
    <span class="stat-label">Grad Accum</span>
  </div>
  <div class="stat-card stat-lavender">
    <span class="stat-value">1,024</span>
    <span class="stat-label">Effective Batch</span>
  </div>
  <div class="stat-card stat-golden">
    <span class="stat-value">16</span>
    <span class="stat-label">Workers</span>
  </div>
</div>

---

**Next:** [Model Architecture →](../architecture/overview.md)
