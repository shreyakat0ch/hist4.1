# Deep-H

Deep-H is a cell-type-aware deep learning framework for predicting histone
modification landscapes from DNA sequence and RNA expression profiles.

## Documentation

For installation, data preparation, model architecture, training, and inference
guides, visit the complete documentation:

**[Open the Deep-H documentation](https://shreyakat0ch.github.io/hist4.1/)**

## Example Inference

A small DNA and RNA example is included in
[`examples/inference`](examples/inference). Run it with:

```bash
python predict.py \
  --dna examples/inference/dna_sequence.fa \
  --rna examples/inference/rna_expression.tsv \
  --weights runs/best_model.pt \
  --output example_results
```

The model checkpoint is not included because it is about 190 MB. Distribute it
through Git LFS or a GitHub Release.

### Visualize Predictions Against ChIP-seq

For a prepared cell line with genome cache, RNA map, and BED ground truth:

```bash
python predict_and_compare.py \
  --cell K562 \
  --chrom chr1 \
  --start 1000000 \
  --end 1100000 \
  --output comparison_results
```

This creates genome tracks, a peak-overlap summary, TSV results, and BED files.
See the [Inference documentation](https://shreyakat0ch.github.io/hist4.1/inference/)
for instructions on reading each visualization.

## Highlights

- Predicts H3K27ac, H3K4me3, H3K27me3, and H3K9me3.
- Combines 32,768 bp DNA windows with RNA expression features.
- Produces scalar intensity and spatial track predictions.
- Uses a cell-conditioned bidirectional Mamba architecture.

## Quick Start

```bash
git clone https://github.com/shreyakat0ch/hist4.1.git
cd hist4.1
pip install -r requirements.txt
python train.py
```
