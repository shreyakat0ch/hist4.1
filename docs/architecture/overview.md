# Model Architecture - Overview

Deep-H's architecture is a **multi-modal fusion network** that combines genomic DNA features with cell-type RNA expression to predict histone modifications. The model is defined in `model.py` as the `ChromaRegressor` class.

![Deep-H Architecture](../assets/images/Presentation3.png)

## Architecture Summary

The model processes two inputs through specialized encoders and fuses them at multiple stages:

```mermaid
flowchart TB
    subgraph inputs ["📥 Inputs"]
        DNA["🧬 DNA Sequence<br/>[B, 4, 32768]"]
        RNA["📊 RNA Expression<br/>[B, 4000]"]
    end

    subgraph dna_enc ["🔵 DNA Encoder"]
        CNN_A["Branch A: Motif<br/>(k=15)"]
        CNN_B["Branch B: Nucleosome<br/>(k=15, s=8)"]
        CNN_C["Branch C: Domain<br/>(k=31)"]
        MERGE["Merge → [B, 256, 512]"]
    end

    subgraph rna_enc ["🟢 RNA Encoder"]
        MLP["MLP: 4000→1024→512→512"]
    end

    subgraph ssm ["🟣 Cell-Conditioned BiMamba"]
        FILM["FiLM γ,β<br/>(RNA Injection #1)"]
        FWD["Forward Mamba-2"]
        BWD["Backward Mamba-2"]
    end

    subgraph xattn ["🟡 Cross-Attention"]
        CA["DNA queries × RNA tokens<br/>(RNA Injection #2)"]
    end

    subgraph outputs ["📤 Outputs"]
        TRACK["📍 Track Head<br/>[B, 4, 512]"]
        SCALAR["🎯 Scalar Head<br/>[B, 4]"]
    end

    DNA --> CNN_A & CNN_B & CNN_C
    CNN_A & CNN_B & CNN_C --> MERGE
    RNA --> MLP
    MERGE --> FILM
    MLP --> FILM
    FILM --> FWD & BWD
    FWD & BWD --> CA
    MLP --> CA
    CA --> TRACK
    CA --> SCALAR
    MLP -.->|"RNA Injection #3"| SCALAR

    style DNA fill:#FFE0E0,stroke:#FF6B6B,color:#333
    style RNA fill:#D4F5F2,stroke:#4ECDC4,color:#333
    style MERGE fill:#DBEAFE,stroke:#60A5FA,color:#333
    style MLP fill:#D1FAE5,stroke:#34D399,color:#333
    style FILM fill:#EDE9FE,stroke:#A78BFA,color:#333
    style FWD fill:#EDE9FE,stroke:#A78BFA,color:#333
    style BWD fill:#EDE9FE,stroke:#A78BFA,color:#333
    style CA fill:#FFF8D6,stroke:#FFD93D,color:#333
    style TRACK fill:#FFF3E0,stroke:#FDBA74,color:#333
    style SCALAR fill:#FFE4E6,stroke:#FB7185,color:#333
```

## The Three RNA Injections

Deep-H's key innovation is injecting cell-type identity at **three distinct levels**, each serving a different purpose:

<div class="feature-grid">
  <div class="feature-card coral">
    <span class="feature-icon">①</span>
    <h3>Per-Layer FiLM Conditioning</h3>
    <p><strong>Where:</strong> Before each of the 4 BiMamba layers.<br/>
    <strong>How:</strong> RNA → γ, β → affine transform on DNA features.<br/>
    <strong>Why:</strong> Applies the <em>same</em> cell-type modulation to every DNA position - teaches the model coarse cell-type-specific patterns.</p>
    <span class="chip chip-coral">Global modulation</span>
  </div>
  <div class="feature-card turquoise">
    <span class="feature-icon">②</span>
    <h3>Cross-Attention</h3>
    <p><strong>Where:</strong> After all BiMamba layers.<br/>
    <strong>How:</strong> Each DNA position <em>independently queries</em> 8 RNA pseudo-tokens.<br/>
    <strong>Why:</strong> A CpG island position can attend to developmental genes; a repeat position to heterochromatin genes. Genuine position × cell-type interaction.</p>
    <span class="chip chip-turquoise">Position-specific</span>
  </div>
  <div class="feature-card lavender">
    <span class="feature-icon">③</span>
    <h3>Head Concatenation</h3>
    <p><strong>Where:</strong> At each regression head.<br/>
    <strong>How:</strong> RNA embedding concatenated with pooled DNA features → MLP.<br/>
    <strong>Why:</strong> Direct shortcut ensuring cell-type identity reaches the final prediction even if upstream conditioning is subtle.</p>
    <span class="chip chip-lavender">Direct shortcut</span>
  </div>
</div>

## Component Summary

| Component | Parameters | Input → Output | Purpose |
|-----------|-----------|----------------|---------|
| Multi-Resolution CNN | ~1.2M | [4, 32768] → [256, 512] | Multi-scale DNA feature extraction |
| RNA MLP | ~4.8M | [4000] → [512] | Cell-type identity compression |
| FiLM Layers (×4) | ~0.5M | [512] → γ, β per layer | Global cell conditioning |
| BiMamba-2 (×4) | ~5.4M | [512, 256] → [512, 256] | Bidirectional sequence modeling |
| Cross-Attention | ~0.8M | DNA × RNA → [512, 256] | Position-specific conditioning |
| Scalar Heads (×4) | ~1.1M | [512] → [4] | Per-mark intensity prediction |
| Track Head | ~1.0M | [256, 512] → [4, 512] | Spatial peak localization |
| **Total** | **~14.8M** | | |

## Detailed Component Pages

Explore each component in depth:

- **[DNA Encoder →](dna_encoder.md)** - Multi-resolution CNN stem
- **[RNA Encoder →](rna_encoder.md)** - MLP compression of gene expression
- **[BiMamba SSM →](bimamba.md)** - Bidirectional state space model with FiLM
- **[Prediction Heads →](heads.md)** - Scalar and track output heads
