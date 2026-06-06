# Installation

This guide covers everything you need to set up HERON from scratch on a machine with a CUDA-capable GPU.

## System Requirements

<div class="feature-grid">
  <div class="feature-card coral">
    <span class="feature-icon">🖥️</span>
    <h3>GPU</h3>
    <p>NVIDIA GPU with <strong>≥24 GB VRAM</strong> (A100, A6000, or RTX 4090 recommended). Training uses mixed-precision (FP16) via PyTorch AMP.</p>
  </div>
  <div class="feature-card turquoise">
    <span class="feature-icon">💾</span>
    <h3>RAM</h3>
    <p><strong>≥64 GB</strong> system RAM recommended. The genome cache and 440 cell-line peak indexes consume significant memory.</p>
  </div>
  <div class="feature-card lavender">
    <span class="feature-icon">💿</span>
    <h3>Storage</h3>
    <p><strong>≥100 GB</strong> free disk space for raw data (ChIP-seq BED files, FASTA genome, RNA-seq expression matrices).</p>
  </div>
  <div class="feature-card golden">
    <span class="feature-icon">🐍</span>
    <h3>Python</h3>
    <p><strong>Python 3.10+</strong> with CUDA 11.8 or 12.x. Conda/Mamba environment recommended for clean dependency management.</p>
  </div>
</div>

## Step 1 — Create a Conda Environment

!!! tip "Why Conda?"
    The `mamba-ssm` package requires CUDA compilation and works best in an isolated environment.

```bash
# Create and activate a new environment
conda create -n heron python=3.10 -y
conda activate heron

# Install CUDA toolkit if not already present
conda install -c conda-forge cudatoolkit=11.8 -y
```

## Step 2 — Install PyTorch

Install PyTorch with CUDA support matching your system's CUDA version:

=== "CUDA 11.8"

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

=== "CUDA 12.1"

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

Verify your installation:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

!!! warning "CUDA Verification"
    If `torch.cuda.is_available()` returns `False`, your CUDA toolkit or GPU driver may need updating. HERON requires a working GPU for training.

## Step 3 — Install Mamba-2 (State Space Model)

Mamba-2 is the core sequence model used in HERON's bidirectional encoder:

```bash
pip install mamba-ssm
```

!!! note "Compilation Time"
    `mamba-ssm` compiles CUDA kernels during installation. This may take **5–10 minutes** and requires a working `nvcc` (NVIDIA CUDA compiler). If compilation fails, ensure your CUDA toolkit matches your PyTorch CUDA version.

## Step 4 — Install Remaining Dependencies

```bash
pip install numpy pandas scikit-learn tqdm pyfaidx
```

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0 | Deep learning framework |
| `mamba-ssm` | ≥2.0 | Mamba-2 selective state space model |
| `numpy` | ≥1.24 | Array operations |
| `pandas` | ≥1.5 | Data manipulation |
| `scikit-learn` | ≥1.2 | Evaluation metrics (AUPRC, MCC, F1) |
| `tqdm` | ≥4.64 | Progress bars |
| `pyfaidx` | ≥0.7 | Fast indexed FASTA access |

## Step 5 — Prepare Reference Data

HERON requires two external reference files:

### 5a. Human Genome (hg38)

```bash
# Download hg38 FASTA
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz

# Index it (required by pyfaidx)
samtools faidx hg38.fa
```

### 5b. Gene Annotation (GTF)

```bash
# Download UCSC knownGene annotation
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/hg38.knownGene.gtf.gz
gunzip hg38.knownGene.gtf.gz
```

### 5c. Set Paths

Edit `config.py` or set environment variables:

```bash
export HISTONE_FASTA_PATH="/path/to/hg38.fa"
export HISTONE_GTF_PATH="/path/to/hg38.knownGene.gtf"
export HISTONE_DATA_DIR="/path/to/chip_seq_data"
```

## Step 6 — Prepare ChIP-seq Data

HERON trains on ChIP-seq BED files from the ENCODE project. The data directory should be structured as:

```
data/
├── cell_line_1/
│   ├── H3K27ac.bed
│   ├── H3K4me3.bed
│   ├── H3K27me3.bed
│   └── H3K9me3.bed
├── cell_line_2/
│   ├── H3K27ac.bed
│   └── H3K4me3.bed       ← not all marks required
└── ...                     (440 cell lines)
```

!!! info "Missing Marks Are Handled"
    HERON uses a **mask-based loss** — cell lines that are missing certain histone marks are automatically excluded from those marks' loss computation. You do not need complete data for all 4 marks.

## Step 7 — Run Preprocessing

```bash
# Build the cell peak index and RNA maps
python data_pipeline.py
```

This creates:

- `processed/cell_peak_indexes.pkl` — Indexed peak locations for all cell lines
- `processed/rna_map.pkl` — Cell-line RNA expression vectors
- `processed/cell_order.json` — Canonical ordering for train/val/test split
- `processed/genome_hg38.bin` — Binary genome cache for fast sequence lookup

## Verify Installation

```bash
# Quick smoke test — trains for 1 batch
python smoke_train.py
```

!!! example "Expected Output"
    ```
    Using device: cuda
    Splits: 352 train, 44 val, 44 test
    [train] Loading peak indexes...
    Training batch completed successfully!
    ```

---

**Next:** [Data Pipeline →](data/overview.md)
