"""
predict.py - Predict histone modification levels from user-provided DNA + RNA.

Usage:
    python predict.py --dna sequence.fa --rna expression.tsv
    python predict.py --dna sequence.fa --rna expression.tsv --output results/
    python predict.py --dna "ATCGATCG..." --rna expression.tsv --weights best_model.pt

Outputs:
    1. Terminal: coloured bar chart + chromatin state interpretation
    2. PNG figure: horizontal bar chart of predicted signal levels
    3. CSV file: machine-readable predictions
    4. JSON file: full results with chromatin state annotation
"""

import argparse
import os
import sys
import json
import pickle
import torch
import numpy as np
import pandas as pd

import config
from model import ChromaRegressor

# ── Mark metadata ────────────────────────────────────────────────────────────
# Each mark has a biological role, colour for plotting, and category.
MARK_INFO = {
    "H3K27ac":  {"role": "Active enhancers + promoters",      "color": "#2ecc71", "category": "activating"},
    "H3K4me3":  {"role": "Active promoters (sharp peaks)",    "color": "#27ae60", "category": "activating"},
    "H3K27me3": {"role": "Polycomb repression (silencing)",   "color": "#e74c3c", "category": "repressive"},
    "H3K9me3":  {"role": "Constitutive heterochromatin",      "color": "#c0392b", "category": "repressive"},
}

# ── Chromatin state rules ────────────────────────────────────────────────────
# Ordered from most specific to least specific.  First match wins.
# Each rule is (name, description, condition_function).

def _classify_chromatin_state(vals):
    """Classify the chromatin state from a dict {mark_name: signal_value}."""
    threshold = config.PRESENCE_THRESHOLD

    ac   = vals.get("H3K27ac", 0)
    me3  = vals.get("H3K4me3", 0)
    k27  = vals.get("H3K27me3", 0)
    k9   = vals.get("H3K9me3", 0)

    ac_on   = ac   > threshold
    me3_on  = me3  > threshold
    k27_on  = k27  > threshold
    k9_on   = k9   > threshold

    # Bivalent promoter
    if me3_on and k27_on and not ac_on:
        return ("Bivalent Promoter", "Carries both activating and repressive marks. Poised.", "#9b59b6", "POISED", "LOW")

    # Active promoter
    if me3_on and ac_on:
        return ("Active Promoter", "Strong H3K4me3 and H3K27ac signal.", "#2ecc71", "OPEN", "HIGH")

    # Active enhancer
    if ac_on and not me3_on:
        return ("Active Enhancer", "H3K27ac marks an active enhancer element.", "#3498db", "OPEN", "HIGH")

    # Polycomb repressed
    if k27_on and not ac_on and not me3_on:
        return ("Polycomb Repressed", "H3K27me3 indicates Polycomb-mediated silencing.", "#e74c3c", "CLOSED", "RESTRICTED")

    # Heterochromatin
    if k9_on:
        return ("Heterochromatin", "H3K9me3 marks constitutive heterochromatin.", "#c0392b", "CLOSED", "RESTRICTED")

    # Weak / mixed
    if any([ac_on, me3_on]):
        return ("Weak Active", "Some activating marks are present but weak.", "#95a5a6", "PARTIALLY OPEN", "MODERATE")

    return ("Quiescent", "No histone marks are significantly enriched.", "#bdc3c7", "CLOSED", "RESTRICTED")


# ── Input processing ─────────────────────────────────────────────────────────

def process_dna(dna_input):
    """Read DNA from a FASTA file or raw string.  Returns uint8 array."""
    if os.path.isfile(dna_input):
        with open(dna_input, "r") as f:
            lines = f.readlines()
        seq = "".join(l.strip() for l in lines if not l.startswith(">"))
    else:
        seq = dna_input

    seq = seq.upper()

    target_len = config.WINDOW_SIZE
    if len(seq) < target_len:
        pad = target_len - len(seq)
        left = pad // 2
        right = pad - left
        print(f"  DNA length {len(seq)} < {target_len}: padding with {pad} Ns "
              f"({left} left, {right} right)")
        seq = ("N" * left) + seq + ("N" * right)
    elif len(seq) > target_len:
        start = (len(seq) - target_len) // 2
        print(f"  DNA length {len(seq)} > {target_len}: centre-cropping to {target_len} bp")
        seq = seq[start : start + target_len]

    # Vectorised ASCII → uint8 encoding: A=0, C=1, G=2, T=3, N=4
    seq_bytes = np.frombuffer(seq.encode("ascii"), dtype="uint8")
    arr = np.full(len(seq_bytes), 4, dtype="uint8")
    arr[seq_bytes == 65] = 0  # A
    arr[seq_bytes == 67] = 1  # C
    arr[seq_bytes == 71] = 2  # G
    arr[seq_bytes == 84] = 3  # T
    return arr


def process_rna(rna_path):
    """Read an RNA-seq TSV and extract the 4000-gene fingerprint vector."""
    if not os.path.exists(rna_path):
        raise FileNotFoundError(f"RNA file not found: {rna_path}")

    rna_map_path = os.path.join(config.PROCESSED_DIR, "rna_map.pkl")
    example_genes_path = os.path.join(
        config.CONFIG_DIR, "examples", "inference", "top_genes.txt"
    )

    if os.path.exists(rna_map_path):
        with open(rna_map_path, "rb") as f:
            rna_data = pickle.load(f)
            top_genes = list(rna_data["top_genes"])
    elif os.path.exists(example_genes_path):
        with open(example_genes_path, "r") as f:
            top_genes = [line.strip() for line in f if line.strip()]
        print(f"  RNA: using bundled gene order from {example_genes_path}")
    else:
        raise FileNotFoundError(
            f"Missing {rna_map_path} and {example_genes_path}. Run "
            "data_pipeline.py or restore the bundled top_genes.txt file."
        )

    if len(top_genes) != config.RNA_INPUT_DIM:
        raise ValueError(
            f"Expected {config.RNA_INPUT_DIM} genes, found {len(top_genes)} "
            "in the gene-order file."
        )

    df = pd.read_csv(rna_path, sep="\t", index_col=0)
    numeric_cols = df.select_dtypes(include=[np.number])
    if numeric_cols.empty:
        raise ValueError("No numeric columns found in the RNA TSV file.")

    avg_tpm = numeric_cols.mean(axis=1)
    log_tpm = np.log2(avg_tpm.values.astype(np.float32) + 1.0)
    log_series = pd.Series(log_tpm, index=avg_tpm.index)

    rna_feat = log_series.reindex(top_genes).fillna(0).values.astype(np.float32)

    matched = log_series.index.isin(top_genes).sum()
    print(f"  RNA: matched {matched}/{len(top_genes)} expected genes "
          f"({matched * 100 / len(top_genes):.1f}%)")

    return rna_feat


# ── Terminal visualisation ───────────────────────────────────────────────────

def _terminal_bar(value, max_val, width=30):
    """Return a text bar like ████████░░░░░░░."""
    if max_val <= 0:
        max_val = 1.0
    fill = int(round(min(value / max_val, 1.0) * width))
    fill = max(0, fill)
    return "█" * fill + "░" * (width - fill)


def print_terminal_results(mark_values, state_name, state_desc, state_color, accessibility, tf_binding):
    """Pretty-print results to the terminal."""
    max_val = max(mark_values.values()) if mark_values else 1.0
    max_val = max(max_val, 1.0)  # avoid division by zero

    print("\n" + "=" * 72)
    print("  HISTONE MARK PREDICTIONS")
    print("=" * 72)

    for mark in config.TARGET_MARKS:
        val = mark_values[mark]
        info = MARK_INFO.get(mark, {})
        cat = info.get("category", "unknown")
        role = info.get("role", "")
        bar = _terminal_bar(val, max_val)
        presence = "PRESENT" if val > config.PRESENCE_THRESHOLD else "absent"

        tag = "+" if cat == "activating" else "-"
        print(f"  [{tag}] {mark:<10s} {bar}  {val:>6.3f}  {presence:<8s}  {role}")

    print("-" * 72)
    print(f"\n  Predicted Chromatin State:  {state_name}")
    print(f"  {state_desc}")
    print(f"\n  ▶ Chromatin Accessibility:  {accessibility}")
    print(f"  ▶ TF Binding Potential:     {tf_binding}")
    print("=" * 72 + "\n")


# ── Matplotlib figure ────────────────────────────────────────────────────────

def save_bar_chart(mark_values, state_name, output_path):
    """Save a horizontal bar chart as a PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not installed - skipping PNG chart.")
        return

    marks = list(config.TARGET_MARKS)
    values = [mark_values[m] for m in marks]
    colors = [MARK_INFO.get(m, {}).get("color", "#95a5a6") for m in marks]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(marks))

    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(marks, fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Predicted Signal  [ log₂(signal + 1) ]", fontsize=11)
    ax.set_title(f"Histone Mark Predictions  →  {state_name}", fontsize=14,
                 fontweight="bold")

    # Threshold line
    ax.axvline(x=config.PRESENCE_THRESHOLD, color="#e74c3c", linestyle="--",
               linewidth=1, alpha=0.7, label=f"Presence threshold ({config.PRESENCE_THRESHOLD})")

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)

    # Legend
    act_patch = mpatches.Patch(color="#2ecc71", label="Activating mark")
    rep_patch = mpatches.Patch(color="#e74c3c", label="Repressive mark")
    ax.legend(handles=[act_patch, rep_patch, ax.get_lines()[0]], loc="lower right",
              fontsize=9)

    ax.set_xlim(0, max(values) * 1.25 + 0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved bar chart → {output_path}")


# ── File outputs ─────────────────────────────────────────────────────────────

def save_csv(mark_values, state_name, output_path):
    """Write predictions to a CSV file."""
    rows = []
    for mark in config.TARGET_MARKS:
        val = mark_values[mark]
        info = MARK_INFO.get(mark, {})
        rows.append({
            "mark": mark,
            "predicted_signal": round(val, 4),
            "present": val > config.PRESENCE_THRESHOLD,
            "category": info.get("category", ""),
            "role": info.get("role", ""),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved CSV       → {output_path}")


def save_json(mark_values, state_name, state_desc, accessibility, tf_binding, output_path):
    """Write predictions + chromatin state to a JSON file."""
    result = {
        "predictions": {},
        "chromatin_state": {
            "name": state_name,
            "description": state_desc,
            "accessibility": accessibility,
            "tf_binding_potential": tf_binding,
        },
        "presence_threshold": config.PRESENCE_THRESHOLD,
        "window_size_bp": config.WINDOW_SIZE,
        "model_marks": config.TARGET_MARKS,
    }
    for mark in config.TARGET_MARKS:
        val = float(mark_values[mark])
        info = MARK_INFO.get(mark, {})
        result["predictions"][mark] = {
            "signal": round(val, 4),
            "present": val > config.PRESENCE_THRESHOLD,
            "category": info.get("category", ""),
            "role": info.get("role", ""),
        }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved JSON      → {output_path}")


# ── Main prediction function ────────────────────────────────────────────────

def predict(dna_input, rna_path, weights_path=None, output_dir=None, use_cpu=False):
    """Run prediction and produce all outputs.

    Parameters
    ----------
    dna_input : str
        Raw DNA sequence string OR path to a FASTA file.
    rna_path : str
        Path to an RNA-seq TSV file (gene × sample TPM matrix).
    weights_path : str, optional
        Path to a trained model checkpoint.  Defaults to runs/best_model.pt.
    output_dir : str, optional
        Directory to save output files.  Defaults to runs/predictions/.
    use_cpu : bool, optional
        Force execution on CPU.
    """
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if weights_path is None:
        weights_path = os.path.join(config.RUNS_DIR, "best_model.pt")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}.\n"
            "  Train the model first (python train.py) or pass --weights."
        )

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\n[1/4] Loading model from {weights_path} ...")
    model = ChromaRegressor().to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch_info = checkpoint.get("epoch", "?")
    corr_info = checkpoint.get("mean_corr", "?")
    print(f"  Checkpoint epoch {epoch_info}, mean Pearson {corr_info}")

    # ── Load normalisation stats ─────────────────────────────────────────
    stats_path = os.path.join(config.PROCESSED_DIR, "norm_stats.json")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(
            config.CONFIG_DIR, "examples", "inference", "norm_stats.json"
        )

    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            d = json.load(f)
        target_mean = np.array(d["mean"], dtype=np.float32)
        target_std = np.array(d["std"], dtype=np.float32)
    else:
        print("  Warning: norm_stats.json not found - outputting raw values.")
        target_mean = np.zeros(config.NUM_MARKS, dtype=np.float32)
        target_std = np.ones(config.NUM_MARKS, dtype=np.float32)

    # ── Process inputs ───────────────────────────────────────────────────
    print("\n[2/4] Processing inputs ...")
    dna_arr = process_dna(dna_input)
    rna_arr = process_rna(rna_path)

    dna_tensor = torch.from_numpy(dna_arr).unsqueeze(0).to(device)
    rna_tensor = torch.from_numpy(rna_arr).unsqueeze(0).to(device)

    # ── Inference ────────────────────────────────────────────────────────
    print("\n[3/4] Running inference ...")
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                pred, track_logits = model(dna_tensor, rna_tensor)
        else:
            pred, track_logits = model(dna_tensor, rna_tensor)

    pred = pred.cpu().numpy()[0]
    track_probs = torch.sigmoid(track_logits[0]).cpu().numpy()  # [NUM_MARKS, 512]

    # Un-normalise to log2(signal + 1) scale
    pred_unnorm = pred * target_std + target_mean
    pred_unnorm = np.maximum(pred_unnorm, 0.0)

    # Build mark → value dict and peak dict
    mark_values = {}
    mark_peaks = {}
    for i, mark in enumerate(config.TARGET_MARKS):
        mark_values[mark] = float(pred_unnorm[i])
        
        # Track parsing
        probs = track_probs[i]
        peak_bins = probs > 0.5
        
        peaks = []
        in_peak = False
        start_bin = 0
        for b_idx in range(len(peak_bins)):
            if peak_bins[b_idx] and not in_peak:
                in_peak = True
                start_bin = b_idx
            elif not peak_bins[b_idx] and in_peak:
                in_peak = False
                end_bin = b_idx - 1
                peaks.append({
                    "start": start_bin * config.TRACK_BIN_SIZE,
                    "end": (end_bin + 1) * config.TRACK_BIN_SIZE,
                    "confidence": float(probs[start_bin:end_bin+1].mean())
                })
        if in_peak:
            peaks.append({
                "start": start_bin * config.TRACK_BIN_SIZE,
                "end": len(peak_bins) * config.TRACK_BIN_SIZE,
                "confidence": float(probs[start_bin:].mean())
            })
        mark_peaks[mark] = peaks

    # ── Classify chromatin state ─────────────────────────────────────────
    state_name, state_desc, state_color, accessibility, tf_binding = _classify_chromatin_state(mark_values)

    # ── Output ───────────────────────────────────────────────────────────
    print("\n[4/4] Generating outputs ...")

    # Terminal
    print_terminal_results(mark_values, state_name, state_desc, state_color, accessibility, tf_binding)

    # Print peaks
    print("\n  DETECTED PEAK COORDINATES")
    print("=" * 72)
    for mark in config.TARGET_MARKS:
        if not mark_peaks[mark]:
            continue
        print(f"  {mark}:")
        for pk in mark_peaks[mark]:
            print(f"    ▶ bp {pk['start']:>5} - {pk['end']:<5}  (conf: {pk['confidence']:.2f})")
    print("=" * 72 + "\n")

    # File outputs
    if output_dir is None:
        output_dir = os.path.join(config.RUNS_DIR, "predictions")
    os.makedirs(output_dir, exist_ok=True)

    save_bar_chart(mark_values, state_name,
                   os.path.join(output_dir, "prediction_chart.png"))
    save_csv(mark_values, state_name,
             os.path.join(output_dir, "prediction.csv"))
    save_json(mark_values, state_name, state_desc, accessibility, tf_binding,
              os.path.join(output_dir, "prediction.json"))

    print(f"\n  All outputs saved to {output_dir}/")
    return mark_values, state_name


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict histone marks from user-provided DNA + RNA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict from a FASTA file and RNA TSV:
  python predict.py --dna region.fa --rna cell_rna.tsv

  # Predict from a raw DNA string:
  python predict.py --dna "ATCGATCG..." --rna cell_rna.tsv

  # Use a specific checkpoint and output directory:
  python predict.py --dna region.fa --rna cell_rna.tsv \\
                    --weights runs/best_model.pt --output results/
        """,
    )
    parser.add_argument("--dna", required=True,
                        help="DNA sequence string OR path to a FASTA file.")
    parser.add_argument("--rna", required=True,
                        help="Path to an RNA-seq TSV file (gene × sample matrix).")
    parser.add_argument("--weights", default=None,
                        help="Path to model checkpoint (default: runs/best_model.pt).")
    parser.add_argument("--output", default=None,
                        help="Directory to save outputs (default: runs/predictions/).")
    parser.add_argument("--cpu", action="store_true",
                        help="Force execution on CPU only.")

    args = parser.parse_args()

    try:
        predict(args.dna, args.rna, args.weights, args.output, args.cpu)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
