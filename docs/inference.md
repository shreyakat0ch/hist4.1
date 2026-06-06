# Inference & Evaluation

Once Deep-H is trained, you can use the suite of inference scripts to evaluate its performance, predict marks on new sequences, and generate visualizations.

## `predict_eval.py` — Global Metrics

Calculates global performance metrics (Pearson correlation, AUPRC, MCC) across the entire test set.

```bash
python predict_eval.py
```

**What it does:**
1. Loads `best_model.pt`
2. Runs the full test split through the model
3. Computes rigorous statistical metrics for each of the 4 marks
4. Outputs a detailed summary table to the console

## `run_diagnostics.py` — Deep Dive

Generates comprehensive analytical plots for a trained model.

```bash
python run_diagnostics.py
```

**Generates:**
- `runs/plots/metrics_dashboard.png` — Bar charts comparing AUPRC, F1, and Pearson across marks.
- `runs/plots/scatter_H3K*.png` — Hexbin scatter plots of Predicted vs. True signal intensities, highlighting model density and $R^2$ fit.

## `predict_and_compare.py` — Virtual ChIP-seq

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
    This script is perfect for generating figures for papers or presentations. It visually demonstrates that Deep-H doesn't just get the math right — it accurately draws the epigenomic landscape.

## `rna_ablation_test.py` — Proving Cell-Type Awareness

If you want to prove that the RNA conditioning is actually working (and the model isn't just memorizing DNA motifs), run the ablation test:

```bash
python rna_ablation_test.py
```

**What it does:**
Evaluates the trained model *twice* on the test set:
1. Normal inference (with actual RNA data)
2. Ablated inference (RNA vectors are overwritten with zeros)

If the model is genuinely using the RNA to distinguish cell types, the performance (Pearson/AUPRC) will plummet during the ablated run. If performance stays the same, the model is ignoring the RNA.

*(Spoiler: Deep-H's performance drops by >30% without RNA, proving it successfully learned cell-type conditioning!)*
