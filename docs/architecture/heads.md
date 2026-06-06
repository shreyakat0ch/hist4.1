# Prediction Heads - Dual Output Architecture

After the BiMamba and Cross-Attention layers, the 512 DNA tokens (each representing 64 bp) contain rich, cell-type-aware contextual information. Deep-H forks this representation into two independent heads: a **Scalar Head** and a **Track Head**.

## 📍 1. Track Head (Spatial Localization)

The Track Head predicts *where* peaks are located at high resolution (64 bp bins).

Before making predictions, the spatial DNA features receive one final cell-type modulation specifically for localization:

```python
# 1. RNA FiLM gate specifically for the track head
film_weights = self.track_rna_film(rna_proj)      # [B, 512]
gamma, beta = film_weights.chunk(2, dim=-1)       # [B, 256] each

# 2. Modulate spatial features
track_feat = dna_feat * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

# 3. Predict track logits via 1x1 Convolution
# Input: [B, 256, 512] -> Output: [B, 4, 512]
track_logits = self.track_head(track_feat.transpose(1, 2))
```

!!! info "Why 1x1 Convolution?"
    A 1x1 convolution acts as a position-wise linear layer. It independently maps the 256-dimensional feature vector at *each* of the 512 spatial bins into 4 logits (one for each histone mark). There is no pooling - the spatial resolution is perfectly preserved.

## 🎯 2. Scalar Heads (Peak Intensity)

The Scalar Head predicts the overall intensity `log2(signal + 1)` of each mark within the window.

Because different histone marks have vastly different spatial footprints (e.g., H3K4me3 is sharp, H3K27me3 is broad), Deep-H uses **Per-Mark Attention Pooling** rather than a generic global average pool.

### Step A: Positional Encoding

Since the downstream attention pooling is permutation-invariant, we first inject explicit positional information:

```python
# Lets the pooling head know where each token is relative to the center
dna_feat = self.pos_enc(dna_feat)
```

### Step B: Mark-Specific Context Pooling

Each mark has its own learnable query vector that attends only to a mark-specific central window:

| Mark | Context Window | Biological Reason |
|------|----------------|-------------------|
| **H3K4me3** | 4,096 bp | Promoters are highly localized (sharp peaks) |
| **H3K27ac** | 16,384 bp | Enhancers can be diffuse and occur in clusters |
| **H3K9me3** | 16,384 bp | Heterochromatin forms medium-broad domains |
| **H3K27me3** | 32,768 bp | Polycomb repression spans massive domains |

```python
# Example pseudo-code for per-mark pooling
for m_idx, mark_name in enumerate(TARGET_MARKS):
    # 1. Slice DNA features to the mark's specific context width
    start, end = get_context_bounds(mark_name)
    mark_tokens = dna_feat[:, start:end, :]
    
    # 2. Learnable query attends to the sliced tokens
    query = self.mark_queries[m_idx]
    attn_out = self.pool_attns[m_idx](query, key=mark_tokens, value=mark_tokens)
    
    pooled_feats.append(attn_out)
```

### Step C: RNA Concatenation & Regression

As a final safeguard, the original RNA cell-type embedding is directly concatenated to the pooled DNA features before passing through independent MLPs for each mark:

```python
for mark_feat, head in zip(pooled_feats, self.mark_heads):
    # Concatenate DNA pooled features [B, 256] with RNA projection [B, 256]
    combined = torch.cat([mark_feat, rna_proj], dim=1)  # [B, 512]
    
    # Independent MLP per mark
    scalar_outputs.append(head(combined))
```

!!! tip "Independent Heads"
    Using separate MLPs (no shared layers) for the final regression ensures that activating marks (H3K27ac) don't interfere with repressive marks (H3K27me3) during training, preventing gradient conflict.

---

**Next:** [Training Pipeline →](../training/pipeline.md)
