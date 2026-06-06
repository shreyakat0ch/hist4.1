# Loss Functions

Deep-H is a multi-task network predicting 4 histone marks simultaneously. Because some cell lines are missing data for certain marks, the loss functions must dynamically mask out invalid targets.

Deep-H optimizes a **Hybrid Loss** that combines three distinct components.

## 1. Masked Huber Loss (Scalar)

The primary loss for predicting peak intensity `log2(signal + 1)`. 

We use Huber Loss (Smooth L1) instead of MSE because it is less sensitive to extreme outliers. ChIP-seq signals can occasionally have massive, artefactual spikes; Huber acts like MSE for small errors and L1 for large errors, preventing these spikes from dominating the gradient.

```python
# 1. Compute Huber for all elements
huber_loss = F.huber_loss(pred, target, reduction='none') # [B, 4]

# 2. Mask out missing marks and apply per-mark weighting
weighted_huber = huber_loss * mask * mark_weights         # [B, 4]

# 3. Average only over valid predictions
masked_huber = weighted_huber.sum() / (mask.sum() + 1e-8)
```

## 2. Masked Pearson Correlation Loss (Scalar)

Huber loss penalizes absolute distance, but epigenomic models also need to accurately rank peaks (e.g., recognizing that enhancer A is stronger than enhancer B). 

We compute the Pearson correlation between predictions and targets across the batch for each mark, and optimize `1 - Correlation`.

$$ \text{Pearson Loss} = 1 - \frac{\text{Cov}(Y, \hat{Y})}{\sigma_Y \sigma_{\hat{Y}}} $$

!!! tip "Why Batch Correlation?"
    Because correlation is scale-invariant, optimizing it directly forces the model to learn the structural variance of the data, perfectly complementing the absolute-scale penalty of the Huber loss.

## 3. Weighted BCE Loss (Track)

For the spatial track head, we predict a binary mask at 64 bp resolution. Because peaks occupy only ~1–2% of the genome, a standard BCE loss would trivially predict "0" everywhere.

We use `BCEWithLogitsLoss` with a massive **pos_weight** to heavily penalize false negatives (missing a true peak):

```python
# H3K27ac/H3K4me3 (denser): weight = 50.0
# H3K27me3/H3K9me3 (sparser): weight = 100.0
_TRACK_POS_WEIGHT = torch.tensor([50.0, 50.0, 100.0, 100.0])

# Compute weighted BCE
bce_raw = F.binary_cross_entropy_with_logits(
    track_logits, track_target, 
    pos_weight=pos_weight, reduction='none'
)

# Apply validity mask and average
track_loss = (bce_raw * track_valid.unsqueeze(-1)).sum() / ...
```

## Combining the Losses

The final objective function combines all three components using configurable balancing weights:

$$ \text{Total Loss} = \Big( \alpha \cdot \text{Huber} + (1-\alpha) \cdot \text{Pearson} \Big) + \lambda_{\text{track}} \cdot \text{TrackBCE} $$

```python
# config.py
LOSS_ALPHA = 0.4          # 40% Huber, 60% Pearson
TRACK_LOSS_WEIGHT = 0.3   # Track loss is scaled down
```

!!! note "Track Weight Tuning"
    `TRACK_LOSS_WEIGHT` was reduced to `0.3` because the 512-bin track BCE generates massive gradients that can overwhelm the scalar heads. At `0.3`, the track head acts as a powerful auxiliary task, forcing the shared Mamba encoder to learn precise spatial features without degrading the scalar Pearson correlation.

## Per-Mark Weighting

Not all histone marks are equally difficult to learn. Repressive marks (H3K27me3) span massive 30kb+ domains and have low signal-to-noise ratios compared to sharp H3K4me3 promoters.

Deep-H applies a static multiplier to the loss for specific marks to focus network capacity:

```python
# [H3K27ac, H3K4me3, H3K27me3, H3K9me3]
MARK_LOSS_WEIGHTS = [1.0, 1.0, 1.5, 1.0]
```
*(H3K27me3 is up-weighted by 1.5×)*

---

**Next:** [Configuration →](config.md)
