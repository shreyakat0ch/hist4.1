# Deep-H Inference Example

This directory contains one small, ready-to-use input pair:

- `dna_sequence.fa`: one 32,768 bp human DNA window in FASTA format.
- `rna_expression.tsv`: TPM expression values for the model's 4,000 genes.
- `top_genes.txt`: the required gene order used during training.
- `norm_stats.json`: target normalization statistics used for readable outputs.

Run the example from the repository root:

```bash
python predict.py \
  --dna examples/inference/dna_sequence.fa \
  --rna examples/inference/rna_expression.tsv \
  --weights runs/best_model.pt \
  --output example_results
```

The trained checkpoint is not included in this small dataset because it is about
190 MB. Store it with Git LFS or attach it to a GitHub Release, then pass its
local path with `--weights`.

## Input Formats

DNA FASTA:

```text
>sample-name
ACGT...
```

RNA TSV:

```text
gene_id	TPM
RPS4Y1	0.4299891
HLA-DRA	0.4299891
```

DNA shorter than 32,768 bp is padded with `N`; longer DNA is center-cropped.
Missing RNA genes are assigned a value of zero.
