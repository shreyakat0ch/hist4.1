# Inference & Evaluation

Once Deep-H is trained, you can use the suite of inference scripts to evaluate its performance, predict marks on new sequences, and generate visualizations.

## Quick Prediction with the Example Dataset

The repository includes a small inference dataset in `examples/inference/`.
It contains one **32,768 bp DNA sequence** and one **4,000-gene RNA expression
profile** representing the cell-line context.

### Included Files

| File | Description |
|------|-------------|
| `dna_sequence.fa` | DNA sequence for one genomic prediction window |
| `rna_expression.tsv` | Gene-by-sample TPM expression table |
| `top_genes.txt` | Model gene order used to construct the RNA vector |
| `norm_stats.json` | Statistics used to restore prediction scale |

### Run the Example

From the repository root:

```bash
python predict.py \
  --dna examples/inference/dna_sequence.fa \
  --rna examples/inference/rna_expression.tsv \
  --weights runs/best_model.pt \
  --output example_results
```

Use `--cpu` if CUDA is unavailable:

```bash
python predict.py \
  --dna examples/inference/dna_sequence.fa \
  --rna examples/inference/rna_expression.tsv \
  --weights runs/best_model.pt \
  --output example_results \
  --cpu
```

!!! warning "Model checkpoint required"
    The example inputs are small enough for regular GitHub storage, but the
    trained checkpoint is about 190 MB. Publish the checkpoint with
    **Git LFS** or as a **GitHub Release asset**, download it locally, and pass
    its path through `--weights`.

### Prediction Outputs

The command creates:

- `prediction.csv` - predicted signal for each of the four histone marks.
- `prediction.json` - predictions plus chromatin-state interpretation.
- `prediction_chart.png` - a visual comparison of predicted mark intensities.
- Terminal output listing detected peak intervals within the 32,768 bp window.

## Use Your Own DNA and RNA

DNA should be supplied as FASTA. Deep-H expects 32,768 bp; shorter sequences are
padded with `N`, and longer sequences are center-cropped.

```text
>my_region
ACGTACGTACGT...
```

RNA should be a tab-separated gene expression table with gene identifiers in
the first column and one or more numeric TPM columns:

```text
gene_id	TPM
RPS4Y1	1.25
HLA-DRA	8.40
COL3A1	3.10
```

The predictor averages multiple numeric sample columns, applies
`log2(TPM + 1)`, orders the genes using `top_genes.txt`, and fills missing genes
with zero.

```bash
python predict.py \
  --dna path/to/region.fa \
  --rna path/to/cell_line_expression.tsv \
  --weights path/to/best_model.pt \
  --output path/to/results
```

## `predict_and_compare.py` - Virtual ChIP-seq

This script scans a genomic region with overlapping 32,768 bp windows, stitches
the predictions together, and compares predicted peaks against ChIP-seq BED
ground truth.

### Requirements

Unlike the lightweight `predict.py` example, this workflow requires:

- `runs/best_model.pt` - trained model checkpoint.
- `processed/rna_map.pkl` - RNA vectors for known cell lines.
- `processed/genome_hg38.bin` - prepared hg38 genome cache.
- ChIP-seq BED files under `DATA_DIR/<cell>/chip_histone/`.

### Choose a Region

Specify an exact interval:

```bash
python predict_and_compare.py \
  --cell K562 \
  --chrom chr1 \
  --start 1000000 \
  --end 1100000 \
  --output comparison_results
```

Use one 32,768 bp window around a center coordinate:

```bash
python predict_and_compare.py \
  --cell K562 \
  --chrom chr1 \
  --center 1050000 \
  --output comparison_results
```

Or let the script select a peak-rich region:

```bash
python predict_and_compare.py \
  --cell K562 \
  --output comparison_results
```

Use `--threshold` to control when a track probability becomes a predicted peak.
The default is `0.3`. A higher value produces fewer, more confident peaks.

```bash
python predict_and_compare.py \
  --cell K562 \
  --chrom chr1 \
  --center 1050000 \
  --threshold 0.5 \
  --output comparison_results
```

### Reading the Genome Track Figure

The main `prediction_<cell>_<chrom>_<start>_<end>.png` figure has one track for
each histone mark:

- **Colored curve and filled area:** predicted peak probability from 0 to 1.
- **Dashed horizontal line:** probability threshold used to call peaks.
- **Gray bars at the top:** true ChIP-seq BED peaks.
- **Green bars at the bottom:** predicted peaks overlapping a true BED peak.
- **Red bars at the bottom:** predicted peaks with no BED overlap.
- **Signal value:** average scalar intensity across the scanned region.
- **State label:** biological interpretation inferred from the mark and signal.

The mark colors are red for H3K27ac, green for H3K4me3, blue for H3K27me3, and
purple for H3K9me3.

### What the Visualization Tells You

The figure answers four separate questions:

1. **Where?** The probability track locates likely modified regions at 64 bp
   resolution.
2. **How strong?** The scalar score estimates overall mark intensity in the
   scanned region.
3. **How accurate?** Green and red peak bars show agreement or disagreement
   with available ChIP-seq ground truth.
4. **What biology is suggested?** Activating marks indicate open regulatory
   chromatin, while repressive marks indicate Polycomb repression or compact
   heterochromatin.

!!! note "Interpretation, not proof"
    The state and TF-binding labels are rule-based hypotheses derived from the
    predicted histone marks. They are useful summaries, not direct TF-binding
    measurements.

### Summary Figure and Data Files

The script also creates:

| Output | Meaning |
|--------|---------|
| `peak_summary_<cell>.png` | True, predicted, and matched peak counts; mean overlap quality; scalar intensity |
| `predictions_<cell>_<chrom>.tsv` | Every predicted peak with confidence, overlap, true coordinates, and biological hypothesis |
| `predicted_<mark>_<cell>.bed` | Predicted intervals for genome-browser loading |

In the summary figure:

- **Peak Detection Summary** compares true and predicted counts; the green
  portion is the number of matched predictions.
- **Prediction-Truth Overlap Quality** reports the mean fraction of each matched
  prediction overlapping a true peak.
- **Overall Signal Intensity** compares regional scalar predictions across the
  four marks.

## `visualizer.py` - Training and Evaluation Figures

`visualizer.py` is a plotting module called automatically by `train.py` and
`test_best_model.py`; it is not a standalone command-line program.

During training, it writes figures under `runs/plots/`, including:

- `dashboard_latest.png` - train/validation loss, Pearson correlation, and
  predicted-versus-actual scatter plots.
- `peak_vs_background.png` - whether predicted intensity separates true peaks
  from background.
- `mark_correlation.png` - predicted co-occurrence between histone marks.
- `calibration.png` - whether predicted and observed signal magnitudes agree.
- `residuals.png` - model error distributions.
- `signal_distributions.png` - actual and predicted validation distributions.

Run training normally to generate these:

```bash
python train.py
```

For held-out evaluation visualizations:

```bash
python test_best_model.py
```

!!! example "Use Case"
    Use `predict_and_compare.py` for genome-browser-style biological examples,
    and use `visualizer.py` outputs to diagnose model training and held-out
    performance.
