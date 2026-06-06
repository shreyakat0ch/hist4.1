# Inference & Evaluation

Once Deep-H is trained, you can use the suite of inference scripts to evaluate its performance, predict marks on new sequences, and generate visualizations.

## `predict_and_compare.py` - Virtual ChIP-seq

Generates beautiful, high-resolution genome browser-style tracks comparing Deep-H's predictions to ground truth across large contiguous genomic regions.

```bash
python predict_and_compare.py
```

**What it does:**
1. Scans a multi-megabase region (e.g., `chr1:1,000,000-2,000,000`) using a sliding window approach.
2. Stitches the 32kb predictions together.
3. Overlays the predicted signal (lines) against the ground truth BED file peaks (shaded areas).
4. Highlights the spatial track head's binary predictions as colored bars.

!!! example "Use Case"
    This script is perfect for generating figures for papers or presentations. It visually demonstrates that Deep-H doesn't just get the math right - it accurately draws the epigenomic landscape.
