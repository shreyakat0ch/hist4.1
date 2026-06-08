"""
Predict & Compare: Combined Track + Signal Prediction with BED Ground Truth Overlap
====================================================================================
Usage:
    python predict_and_compare.py --cell KNS-42 --chrom chr1 --start 1000000 --end 1100000
    python predict_and_compare.py --cell KNS-42 --chrom chr1 --center 1050000
    python predict_and_compare.py --cell KNS-42   # auto-picks a peak-rich region
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import seaborn as sns

# Numpy compat
try:
    if int(np.__version__.split('.')[0]) < 2:
        import numpy.core.numeric as _num; import numpy.core.multiarray as _mul
        sys.modules['numpy._core'] = sys.modules['numpy.core']
        sys.modules['numpy._core.numeric'] = _num; sys.modules['numpy._core.multiarray'] = _mul
except: pass

import config
from model import ChromaRegressor
from genome_cache import GenomeCache
import pickle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MARKS = config.TARGET_MARKS
# Beautiful light-theme color palette
MARK_COLORS = {"H3K27ac": "#E74C3C", "H3K4me3": "#2ECC71", "H3K27me3": "#3498DB", "H3K9me3": "#9B59B6"}
TRACK_THRESHOLD = 0.3

# Set seaborn style for beautiful plots
sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#cccccc", "grid.color": "#ebebeb"})

def load_model():
    model = ChromaRegressor().to(DEVICE)
    ckpt = torch.load("runs/best_model.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')})")
    return model

def load_bed_peaks(cell, mark):
    bed_path = os.path.join(config.DATA_DIR, cell, "chip_histone", f"{mark}.05.bed")
    if not os.path.exists(bed_path):
        return pd.DataFrame(columns=["chrom","start","end","signal"])
    df = pd.read_csv(bed_path, sep="\t", header=None, usecols=[0,1,2,6],
                     names=["chrom","start","end","signal"])
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce").fillna(0)
    return df

def get_peaks_in_region(bed_df, chrom, region_start, region_end):
    if bed_df.empty: return pd.DataFrame(columns=["chrom","start","end","signal"])
    mask = (bed_df["chrom"] == chrom) & (bed_df["start"] < region_end) & (bed_df["end"] > region_start)
    return bed_df[mask].copy()

def find_peak_rich_region(cell):
    print("Auto-finding a peak-rich region...")
    best_chrom, best_center, best_count = "chr1", 1000000, 0
    for mark in MARKS:
        bed = load_bed_peaks(cell, mark)
        if bed.empty: continue
        for _, row in bed.sample(min(500, len(bed))).iterrows():
            center = (row["start"] + row["end"]) // 2
            count = sum(1 for m in MARKS
                       for _, r in get_peaks_in_region(load_bed_peaks(cell, m), row["chrom"],
                                                       center - config.HALF_WINDOW,
                                                       center + config.HALF_WINDOW).iterrows())
            if count > best_count:
                best_count, best_chrom, best_center = count, row["chrom"], center
    print(f"  Best region: {best_chrom}:{best_center - config.HALF_WINDOW}-{best_center + config.HALF_WINDOW} ({best_count} peaks)")
    return best_chrom, best_center - config.HALF_WINDOW, best_center + config.HALF_WINDOW

def predict_region(model, genome, rna, chrom, start, end):
    window = config.WINDOW_SIZE
    step = window // 2
    all_track_probs = {m: [] for m in MARKS}
    all_track_positions = {m: [] for m in MARKS}
    scalars = {m: [] for m in MARKS}

    positions = list(range(start, max(start+1, end - window + 1), step))
    if not positions or positions[-1] + window < end:
        positions.append(max(start, end - window))

    with torch.no_grad():
        for win_start in positions:
            win_end = win_start + window
            dna = genome.get_seq(chrom, win_start, win_end)
            dna_t = torch.from_numpy(dna).unsqueeze(0).to(DEVICE)
            rna_t = torch.from_numpy(rna).float().unsqueeze(0).to(DEVICE)

            scalar_out, track_logits = model(dna_t, rna_t)
            track_probs = torch.sigmoid(track_logits).cpu().numpy()[0]
            scalar_vals = scalar_out.cpu().numpy()[0]

            for m_idx, mark in enumerate(MARKS):
                bins = track_probs[m_idx]
                bin_positions = np.arange(len(bins)) * config.TRACK_BIN_SIZE + win_start
                all_track_probs[mark].append(bins)
                all_track_positions[mark].append(bin_positions)
                scalars[mark].append(scalar_vals[m_idx])

    merged_tracks = {}
    for mark in MARKS:
        pos_to_probs = defaultdict(list)
        for positions_arr, probs_arr in zip(all_track_positions[mark], all_track_probs[mark]):
            for pos, prob in zip(positions_arr, probs_arr):
                if start <= pos < end:
                    pos_to_probs[pos].append(prob)
        sorted_pos = sorted(pos_to_probs.keys())
        merged_pos = np.array(sorted_pos)
        merged_prob = np.array([np.mean(pos_to_probs[p]) for p in sorted_pos])
        merged_tracks[mark] = (merged_pos, merged_prob)

    avg_scalars = {m: float(np.mean(scalars[m])) for m in MARKS}
    return merged_tracks, avg_scalars

def extract_predicted_peaks(positions, probs, threshold):
    peaks = []
    above = probs >= threshold
    if len(above) == 0: return peaks

    in_peak = False
    peak_start = 0
    for i in range(len(above)):
        if above[i] and not in_peak:
            in_peak = True
            peak_start = i
        elif not above[i] and in_peak:
            in_peak = False
            ps = int(positions[peak_start])
            pe = int(positions[i-1]) + config.TRACK_BIN_SIZE
            max_prob = float(probs[peak_start:i].max())
            mean_prob = float(probs[peak_start:i].mean())
            peaks.append({"start": ps, "end": pe, "length": pe - ps,
                         "max_prob": max_prob, "mean_prob": mean_prob})
    if in_peak:
        ps = int(positions[peak_start])
        pe = int(positions[-1]) + config.TRACK_BIN_SIZE
        max_prob = float(probs[peak_start:].max())
        mean_prob = float(probs[peak_start:].mean())
        peaks.append({"start": ps, "end": pe, "length": pe - ps,
                     "max_prob": max_prob, "mean_prob": mean_prob})
    return peaks

def compute_overlap(pred_peaks, true_peaks_df, chrom):
    results = []
    for pp in pred_peaks:
        best_overlap = 0
        best_true = None
        for _, tp in true_peaks_df.iterrows():
            ov_start = max(pp["start"], tp["start"])
            ov_end = min(pp["end"], tp["end"])
            if ov_start < ov_end:
                overlap_bp = ov_end - ov_start
                overlap_frac = overlap_bp / pp["length"]
                if overlap_bp > best_overlap:
                    best_overlap = overlap_bp
                    best_true = tp
                    best_frac = overlap_frac

        if best_true is not None:
            results.append({**pp, "overlap_bp": best_overlap, "overlap_frac": round(best_frac, 3),
                           "true_start": int(best_true["start"]), "true_end": int(best_true["end"]),
                           "true_signal": round(float(best_true["signal"]), 3), "matched": True})
        else:
            results.append({**pp, "overlap_bp": 0, "overlap_frac": 0.0,
                           "true_start": "NA", "true_end": "NA", "true_signal": "NA", "matched": False})
    return results

def get_biological_hypothesis(mark, signal, has_peaks):
    """Generate biological hypotheses based on predicted mark state."""
    if not has_peaks and signal < 1.0:
        return "Inactive/Background", "Low"
    
    if mark == "H3K27ac":
        return "Active Enhancer/Promoter (Open Chromatin)", "High (Activating TFs)"
    elif mark == "H3K4me3":
        return "Active Promoter (Open Chromatin)", "High (Transcription Machinery)"
    elif mark == "H3K27me3":
        return "Polycomb Repressed (Closed Chromatin)", "Low (Repressive TFs)"
    elif mark == "H3K9me3":
        return "Heterochromatin (Highly Compacted)", "Very Low"
    return "Unknown", "Unknown"

def make_comparison_plot(chrom, region_start, region_end, merged_tracks, avg_scalars,
                        all_true_peaks, all_pred_peaks_with_overlap, cell, output_dir):
    fig = plt.figure(figsize=(20, 4 * len(MARKS) + 3), facecolor="#FFFFFF")
    gs = GridSpec(len(MARKS) + 1, 1, height_ratios=[1] * len(MARKS) + [0.5], hspace=0.35)

    region_len = region_end - region_start

    for m_idx, mark in enumerate(MARKS):
        ax = fig.add_subplot(gs[m_idx])
        ax.set_facecolor("#FAFAFA")
        color = MARK_COLORS[mark]

        positions, probs = merged_tracks[mark]
        true_peaks = all_true_peaks[mark]

        if len(positions) > 0:
            ax.fill_between(positions, probs, alpha=0.3, color=color, label="Predicted prob")
            ax.plot(positions, probs, color=color, linewidth=1.2, alpha=0.9)

        ax.axhline(y=TRACK_THRESHOLD, color="#34495E", linewidth=1.0, linestyle="--", alpha=0.4)

        for _, tp in true_peaks.iterrows():
            ax.axvspan(tp["start"], tp["end"], ymin=0.85, ymax=1.0, alpha=0.6,
                      color="#95A5A6", label="True BED peak" if _ == true_peaks.index[0] else "")

        pred_results = all_pred_peaks_with_overlap[mark]
        for j, pr in enumerate(pred_results):
            c = "#27AE60" if pr["matched"] else "#C0392B"
            ax.axvspan(pr["start"], pr["end"], ymin=0.0, ymax=0.12, alpha=0.7, color=c,
                      label=("Pred (match)" if pr["matched"] else "Pred (miss)") if j == 0 else "")

        scalar_val = avg_scalars[mark]
        raw_signal = 2**scalar_val - 1 if scalar_val > 0 else 0
        
        # Add hypothesis to the plot
        chrom_state, tf_bind = get_biological_hypothesis(mark, scalar_val, len(pred_results)>0)
        
        ax.text(0.98, 0.92, f"Signal: {scalar_val:.2f} (raw≈{raw_signal:.1f})\nState: {chrom_state}",
                transform=ax.transAxes, fontsize=10, color="#2C3E50", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", edgecolor=color, alpha=0.9))

        n_pred = len(pred_results)
        n_true = len(true_peaks)
        n_match = sum(1 for p in pred_results if p["matched"])
        ax.text(0.02, 0.92, f"Pred: {n_pred} | True: {n_true} | Matched: {n_match}",
                transform=ax.transAxes, fontsize=10, color="#7F8C8D", ha="left", va="top")

        ax.set_ylabel(mark, fontsize=13, fontweight="bold", color=color)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlim(region_start, region_end)
        ax.tick_params(colors="#2C3E50", labelsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#BDC3C7"); ax.spines["left"].set_color("#BDC3C7")

        if m_idx == 0:
            handles = [mpatches.Patch(color="#95A5A6", alpha=0.6, label="True BED peak"),
                      mpatches.Patch(color="#27AE60", alpha=0.7, label="Pred (matched)"),
                      mpatches.Patch(color="#C0392B", alpha=0.7, label="Pred (no match)")]
            ax.legend(handles=handles, loc="upper center", ncol=3, fontsize=10,
                     framealpha=0.9, facecolor="#FFFFFF", edgecolor="#BDC3C7",
                     labelcolor="#2C3E50", bbox_to_anchor=(0.5, 1.15))

    ax_coord = fig.add_subplot(gs[-1])
    ax_coord.set_facecolor("#FFFFFF")
    ax_coord.set_xlim(region_start, region_end)
    ax_coord.set_xlabel(f"Genomic Position ({chrom})", fontsize=13, color="#2C3E50", fontweight="bold")
    ax_coord.tick_params(colors="#2C3E50", labelsize=10)
    ax_coord.spines["top"].set_visible(False); ax_coord.spines["right"].set_visible(False)
    ax_coord.spines["bottom"].set_color("#BDC3C7"); ax_coord.spines["left"].set_visible(False)
    ax_coord.set_yticks([])

    fig.suptitle(f"Hist4.1 Prediction vs Ground Truth - {cell} | {chrom}:{region_start:,}-{region_end:,}",
                fontsize=18, fontweight="bold", color="#2C3E50", y=0.95)

    plot_path = os.path.join(output_dir, f"prediction_{cell}_{chrom}_{region_start}_{region_end}.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    print(f"Plot saved: {plot_path}")
    return plot_path

def make_peak_detail_plot(chrom, all_pred_peaks_with_overlap, all_true_peaks, avg_scalars, cell, output_dir):
    fig = plt.figure(figsize=(18, 6), facecolor="#FFFFFF")
    gs = GridSpec(1, 3, wspace=0.3)

    # Panel 1: Peak counts
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#FAFAFA")
    x = np.arange(len(MARKS)); w = 0.35
    true_counts = [len(all_true_peaks[m]) for m in MARKS]
    pred_counts = [len(all_pred_peaks_with_overlap[m]) for m in MARKS]
    matched_counts = [sum(1 for p in all_pred_peaks_with_overlap[m] if p["matched"]) for m in MARKS]

    ax1.bar(x - w/2, true_counts, w, label="True BED", color="#BDC3C7", alpha=0.8)
    ax1.bar(x + w/2, pred_counts, w, label="Predicted", color="#E74C3C", alpha=0.7)
    for i, mc in enumerate(matched_counts):
        ax1.bar(x[i] + w/2, mc, w, color="#27AE60", alpha=0.9)

    ax1.set_xticks(x); ax1.set_xticklabels(MARKS, fontsize=11, color="#2C3E50")
    ax1.set_ylabel("Peak Count", color="#2C3E50", fontsize=12)
    ax1.set_title("Peak Detection Summary", color="#2C3E50", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10, facecolor="#FFFFFF", edgecolor="#BDC3C7", labelcolor="#2C3E50")
    ax1.tick_params(colors="#2C3E50")

    # Panel 2: Overlap quality
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#FAFAFA")
    overlap_fracs = []
    for m in MARKS:
        fracs = [p["overlap_frac"] for p in all_pred_peaks_with_overlap[m] if p["matched"]]
        overlap_fracs.append(np.mean(fracs) if fracs else 0)

    colors = [MARK_COLORS[m] for m in MARKS]
    bars = ax2.bar(x, overlap_fracs, 0.6, color=colors, alpha=0.85)
    for bar, val in zip(bars, overlap_fracs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", fontsize=11, color="#2C3E50")

    ax2.set_xticks(x); ax2.set_xticklabels(MARKS, fontsize=11, color="#2C3E50")
    ax2.set_ylabel("Mean Overlap Fraction", color="#2C3E50", fontsize=12)
    ax2.set_title("Prediction-Truth Overlap Quality", color="#2C3E50", fontsize=14, fontweight="bold")
    ax2.set_ylim(0, 1.15)
    ax2.tick_params(colors="#2C3E50")

    # Panel 3: Signal Intensity
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#FAFAFA")
    scalars = [avg_scalars[m] for m in MARKS]
    
    bars3 = ax3.bar(x, scalars, 0.6, color=colors, alpha=0.85)
    for bar, val in zip(bars3, scalars):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", fontsize=11, color="#2C3E50")
        
    ax3.set_xticks(x); ax3.set_xticklabels(MARKS, fontsize=11, color="#2C3E50")
    ax3.set_ylabel("Mean Scalar Intensity (log2)", color="#2C3E50", fontsize=12)
    ax3.set_title("Overall Signal Intensity", color="#2C3E50", fontsize=14, fontweight="bold")
    max_scalar = max(scalars) if scalars else 1.0
    ax3.set_ylim(0, max_scalar * 1.2)
    ax3.tick_params(colors="#2C3E50")

    sns.despine(fig)

    fig.tight_layout()
    path = os.path.join(output_dir, f"peak_summary_{cell}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    print(f"Summary plot saved: {path}")

def save_tsv(chrom, all_pred_peaks_with_overlap, avg_scalars, cell, output_dir):
    rows = []
    for mark in MARKS:
        chrom_state, tf_bind = get_biological_hypothesis(mark, avg_scalars[mark], len(all_pred_peaks_with_overlap[mark]) > 0)
        
        for p in all_pred_peaks_with_overlap[mark]:
            rows.append({
                "cell_line": cell, "mark": mark, "chrom": chrom,
                "pred_start": p["start"], "pred_end": p["end"], "pred_length": p["length"],
                "max_prob": round(p["max_prob"], 4), "mean_prob": round(p["mean_prob"], 4),
                "scalar_intensity_log2": round(avg_scalars[mark], 4),
                "scalar_intensity_raw": round(2**avg_scalars[mark] - 1, 2) if avg_scalars[mark] > 0 else 0,
                "matched_to_bed": p["matched"], "overlap_bp": p["overlap_bp"],
                "overlap_frac": p["overlap_frac"],
                "true_start": p["true_start"], "true_end": p["true_end"],
                "true_signal": p["true_signal"],
                "chromatin_accessibility_hypothesis": chrom_state,
                "tf_binding_potential": tf_bind
            })
            
    if not rows:
        print("No predicted peaks found.")
        return

    df = pd.DataFrame(rows)
    tsv_path = os.path.join(output_dir, f"predictions_{cell}_{chrom}.tsv")
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"TSV saved: {tsv_path} ({len(df)} peaks)")

    for mark in MARKS:
        mark_df = df[df["mark"] == mark]
        if mark_df.empty: continue
        bed_path = os.path.join(output_dir, f"predicted_{mark}_{cell}.bed")
        bed_out = mark_df[["chrom", "pred_start", "pred_end", "max_prob"]].copy()
        bed_out.columns = ["chrom", "start", "end", "score"]
        bed_out.to_csv(bed_path, sep="\t", index=False, header=False)
        print(f"  BED: {bed_path} ({len(bed_out)} peaks)")

def main():
    parser = argparse.ArgumentParser(description="Hist4.1 Prediction & Comparison")
    parser.add_argument("--cell", required=True, help="Cell line name (e.g. K562)")
    parser.add_argument("--chrom", default=None, help="Chromosome (e.g. chr1)")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--center", type=int, default=None, help="Center position (uses WINDOW_SIZE)")
    parser.add_argument("--threshold", type=float, default=TRACK_THRESHOLD)
    parser.add_argument("--output", default="runs/predictions", help="Output directory")
    args = parser.parse_args()

    threshold = args.threshold
    os.makedirs(args.output, exist_ok=True)

    model = load_model()
    genome = GenomeCache(config.FASTA_PATH, config.GENOME_CACHE_PATH)

    with open(os.path.join(config.PROCESSED_DIR, "rna_map.pkl"), "rb") as f:
        rna_data = pickle.load(f)
        rna_map = rna_data["rna_map"]
    if args.cell not in rna_map:
        print(f"ERROR: Cell line '{args.cell}' not found.")
        return
    rna = rna_map[args.cell]

    if args.center:
        chrom = args.chrom or "chr1"
        region_start = args.center - config.HALF_WINDOW
        region_end = args.center + config.HALF_WINDOW
    elif args.start and args.end:
        chrom = args.chrom or "chr1"
        region_start, region_end = args.start, args.end
    else:
        chrom, region_start, region_end = find_peak_rich_region(args.cell)

    print(f"\nPredicting: {args.cell} | {chrom}:{region_start:,}-{region_end:,} ({(region_end-region_start)//1000}kb)")

    merged_tracks, avg_scalars = predict_region(model, genome, rna, chrom, region_start, region_end)

    all_true_peaks = {}
    all_pred_peaks_with_overlap = {}
    for mark in MARKS:
        bed_df = load_bed_peaks(args.cell, mark)
        true_in_region = get_peaks_in_region(bed_df, chrom, region_start, region_end)
        all_true_peaks[mark] = true_in_region

        positions, probs = merged_tracks[mark]
        pred_peaks = extract_predicted_peaks(positions, probs, threshold)
        overlap_results = compute_overlap(pred_peaks, true_in_region, chrom)
        all_pred_peaks_with_overlap[mark] = overlap_results

        n_match = sum(1 for p in overlap_results if p["matched"])
        print(f"  {mark}: {len(pred_peaks)} predicted, {len(true_in_region)} true, {n_match} matched")

    print("\nGenerating outputs...")
    save_tsv(chrom, all_pred_peaks_with_overlap, avg_scalars, args.cell, args.output)
    make_comparison_plot(chrom, region_start, region_end, merged_tracks, avg_scalars,
                        all_true_peaks, all_pred_peaks_with_overlap, args.cell, args.output)
    make_peak_detail_plot(chrom, all_pred_peaks_with_overlap, all_true_peaks, avg_scalars, args.cell, args.output)

    print(f"\n{'='*80}")
    print(f"  PREDICTION SUMMARY & BIOLOGICAL HYPOTHESIS - {args.cell}")
    print(f"{'='*80}")
    for mark in MARKS:
        preds = all_pred_peaks_with_overlap[mark]
        n_match = sum(1 for p in preds if p["matched"])
        avg_ov = np.mean([p["overlap_frac"] for p in preds if p["matched"]]) if n_match else 0
        chrom_state, tf_bind = get_biological_hypothesis(mark, avg_scalars[mark], len(preds)>0)
        
        print(f"  {mark:12s} | Signal={avg_scalars[mark]:.2f} | Match={n_match:3d}/{len(preds):3d} | AvgOverlap={avg_ov:.1%}")
        print(f"               ↳ Hypothesis: {chrom_state} -> {tf_bind} TF Binding")
    print(f"{'='*80}")
    
    # Identify the dominant mark by signal intensity
    dominant_mark = max(avg_scalars, key=avg_scalars.get)
    print(f"\n  ★ Dominant Mark by Signal Intensity: {dominant_mark} ({avg_scalars[dominant_mark]:.2f})")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
