# Data Pipeline — Overview

HERON's data pipeline transforms raw biological data into model-ready tensors. This page provides a high-level overview; subsequent pages dive into each stage.

<div class="diagram-container">
  <img src="../assets/images/data_pipeline.png" alt="Data Construction and Feature Engineering Pipeline" />
  <p class="diagram-caption"><strong>Figure:</strong> Complete data construction and feature engineering pipeline — from raw data sources to model-ready batches</p>
</div>

## Pipeline Stages

The pipeline has **five major stages**, each designed to address specific challenges in epigenomic data:

```mermaid
flowchart LR
    A["📦 Raw Data<br/>Sources"] --> B["🪟 Window<br/>Sampling"]
    B --> C["🔬 Feature<br/>Extraction"]
    C --> D["🔄 Data<br/>Augmentation"]
    D --> E["📊 Model-Ready<br/>Batch"]
    
    style A fill:#FFE0E0,stroke:#FF6B6B,color:#333
    style B fill:#D4F5F2,stroke:#4ECDC4,color:#333
    style C fill:#EDE9FE,stroke:#A78BFA,color:#333
    style D fill:#FFF8D6,stroke:#FFD93D,color:#333
    style E fill:#DBEAFE,stroke:#60A5FA,color:#333
```

<div class="feature-grid">
  <div class="feature-card coral">
    <span class="feature-icon">📦</span>
    <h3>1. Raw Data Sources</h3>
    <p>Three inputs: hg38 reference genome (FASTA), RNA-seq expression per cell line, and ChIP-seq BED files for 4 histone marks across 440 cell lines.</p>
  </div>
  <div class="feature-card turquoise">
    <span class="feature-icon">🪟</span>
    <h3>2. Window Sampling</h3>
    <p>Balanced 50/50 peak/background sampling with smart oversampling for rare marks. 3000 peaks per mark per cell, backgrounds kept 20kb from any peak.</p>
  </div>
  <div class="feature-card lavender">
    <span class="feature-icon">🔬</span>
    <h3>3. Feature Extraction</h3>
    <p>DNA → one-hot [4, 32768]. RNA → top 4000 gene expression vector. Targets → scalar log₂(signal+1) + binary track at 64bp bins.</p>
  </div>
  <div class="feature-card golden">
    <span class="feature-icon">🔄</span>
    <h3>4. Augmentation</h3>
    <p>Random offset ±2000bp for translation invariance + 50% reverse complement flip. Both DNA and track targets are flipped together.</p>
  </div>
</div>

## Data Flow Summary

| Stage | Input | Output | Key Design Choice |
|-------|-------|--------|-------------------|
| Raw Sources | Genome FASTA, BED files, RNA-seq | Indexed data structures | Binary genome cache for O(1) lookup |
| Window Sampling | Peak coordinates, cell list | (cell, chrom, center) tuples | Oversample sparse marks with replacement |
| Feature Extraction | Window coordinates | DNA, RNA, targets, masks | Mask-based loss handles missing marks |
| Augmentation | Raw features | Augmented features | RC flip + offset preserve biology |
| Batching | Augmented features | `[B, 4, 32768]`, `[B, 4000]` | PyTorch DataLoader with 16 workers |

---

**Next:** [Data Construction →](construction.md) · [Feature Engineering →](features.md)
