import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import config

# ── Consistent Color Palette ──────────────────────────────────────────────
MARK_COLORS = {
    'H3K27ac':  '#e74c3c',
    'H3K4me3':  '#2ecc71',
    'H3K27me3': '#3498db',
    'H3K9me3':  '#9b59b6',
}

def _get_color(mark):
    return MARK_COLORS.get(mark, '#95a5a6')

def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ═══════════════════════════════════════════════════════════════════════════
# 1. EPOCH SUMMARY DASHBOARD - single page with all key info
# ═══════════════════════════════════════════════════════════════════════════

def plot_epoch_dashboard(epoch, history, all_preds, all_targets, all_masks,
                         norm_mean, norm_std, output_path):
    marks = config.TARGET_MARKS
    n_marks = len(marks)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'Epoch {epoch+1} - Training Dashboard', fontsize=18, fontweight='bold', y=0.98)
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.35)

    epochs_arr = np.arange(1, len(history['train_loss']) + 1)

    # ── Panel 1: Loss Curves ──
    ax = fig.add_subplot(gs[0, 0:2])
    ax.plot(epochs_arr, history['train_loss'], 'o-', color='#e74c3c', markersize=4, label='Train', linewidth=2)
    ax.plot(epochs_arr, history['val_loss'], 's-', color='#3498db', markersize=4, label='Val', linewidth=2)
    _style_ax(ax, 'Loss Curves', 'Epoch', 'Loss')
    ax.legend(fontsize=9)

    # ── Panel 2: Pearson Correlation Curves ──
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(epochs_arr, history['avg_pearson'], 'D-', color='black', linewidth=2.5, markersize=5, label='Mean', zorder=5)
    for m in marks:
        if m in history['per_mark_pearson']:
            ax.plot(epochs_arr, history['per_mark_pearson'][m], '--', color=_get_color(m), alpha=0.7, label=m)
    _style_ax(ax, 'Pearson Correlation', 'Epoch', 'Pearson r')
    ax.set_ylim(-0.1, 1.0)
    ax.legend(fontsize=7, ncol=2)

    # ── Panel 3: Correlation Bar Chart (current epoch) ──
    ax = fig.add_subplot(gs[0, 3])
    corrs = []
    for m_idx, m in enumerate(marks):
        valid = all_masks[:, m_idx] > 0.5
        if valid.sum() > 2:
            c = np.corrcoef(all_preds[valid, m_idx], all_targets[valid, m_idx])[0, 1]
            corrs.append(c if not np.isnan(c) else 0)
        else:
            corrs.append(0)
    bars = ax.bar(marks, corrs, color=[_get_color(m) for m in marks], edgecolor='white', linewidth=1.5)
    for bar, c in zip(bars, corrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{c:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.05)
    _style_ax(ax, 'Per-Mark Correlation', '', 'Pearson r')
    ax.tick_params(axis='x', rotation=30)

    # ── Panel 4-7: Hexbin Scatter per Mark ──
    for m_idx in range(min(n_marks, 4)):
        ax = fig.add_subplot(gs[1, m_idx])
        valid = all_masks[:, m_idx] > 0.5
        if valid.sum() < 10:
            continue
        p = all_preds[valid, m_idx]
        t = all_targets[valid, m_idx]
        if len(p) > 15000:
            idx = np.random.choice(len(p), 15000, replace=False)
            p, t = p[idx], t[idx]
        hb = ax.hexbin(t, p, gridsize=40, cmap='YlOrRd', mincnt=1)
        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.6, linewidth=1.5)
        corr = np.corrcoef(p, t)[0, 1] if len(p) > 1 else 0
        _style_ax(ax, f'{marks[m_idx]} (r={corr:.3f})', 'Actual (Z)', 'Predicted (Z)')
        plt.colorbar(hb, ax=ax, shrink=0.7, label='Count')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 2. PEAK vs BACKGROUND VIOLIN PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_peak_vs_background(all_preds, all_targets, all_masks,
                            norm_mean, norm_std, output_path):
    marks = config.TARGET_MARKS
    n_marks = len(marks)

    fig, axes = plt.subplots(1, n_marks, figsize=(5 * n_marks, 7))
    if n_marks == 1:
        axes = [axes]

    for m in range(n_marks):
        valid = all_masks[:, m] > 0.5
        if valid.sum() < 10:
            continue

        p_raw = all_preds[valid, m] * norm_std[m] + norm_mean[m]
        t_raw = all_targets[valid, m] * norm_std[m] + norm_mean[m]

        is_peak = t_raw > 0.1
        peak_p = p_raw[is_peak]
        bg_p = p_raw[~is_peak]

        ax = axes[m]
        if len(bg_p) < 2 or len(peak_p) < 2:
            ax.set_title(f'{marks[m]}\n(insufficient data)')
            continue

        # Violin plot
        parts = ax.violinplot([bg_p, peak_p], positions=[1, 2], showmeans=True,
                              showmedians=True, showextrema=False)
        colors = ['#3498db', '#e74c3c']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        parts['cmeans'].set_color('black')
        parts['cmedians'].set_color('white')

        # Cohen's d
        mu_p, mu_b = peak_p.mean(), bg_p.mean()
        std_pool = np.sqrt((peak_p.var() + bg_p.var()) / 2)
        d = (mu_p - mu_b) / max(std_pool, 1e-8)

        # AUROC
        y_true = np.concatenate([np.zeros(len(bg_p)), np.ones(len(peak_p))])
        y_score = np.concatenate([bg_p, peak_p])
        try:
            auroc = roc_auc_score(y_true, y_score)
        except:
            auroc = 0.5

        ax.set_xticks([1, 2])
        ax.set_xticklabels([f'Background\n(n={len(bg_p):,})', f'Peak\n(n={len(peak_p):,})'])
        _style_ax(ax, f'{marks[m]}\nd={d:.2f}  AUROC={auroc:.3f}', '', 'Predicted log₂(sig+1)')

    fig.suptitle('Peak vs Background Discrimination', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 3. MARK CORRELATION HEATMAP - does the model learn co-occurrence?
# ═══════════════════════════════════════════════════════════════════════════

def plot_mark_correlation_heatmap(all_preds, all_masks, output_path):
    marks = config.TARGET_MARKS
    n = len(marks)

    # Use samples where ALL marks are valid
    all_valid = np.all(all_masks > 0.5, axis=1)
    if all_valid.sum() < 10:
        return

    preds_valid = all_preds[all_valid]
    corr_matrix = np.corrcoef(preds_valid.T)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(marks, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(marks, fontsize=10)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=11,
                    fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')
    _style_ax(ax, 'Predicted Mark Co-occurrence', '', '')
    ax.spines['left'].set_visible(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 4. CALIBRATION PLOT - are predicted values trustworthy?
# ═══════════════════════════════════════════════════════════════════════════

def plot_calibration(all_preds, all_targets, all_masks,
                     norm_mean, norm_std, output_path, n_bins=20):
    marks = config.TARGET_MARKS
    n_marks = len(marks)

    fig, axes = plt.subplots(1, n_marks, figsize=(5 * n_marks, 5))
    if n_marks == 1:
        axes = [axes]

    for m in range(n_marks):
        valid = all_masks[:, m] > 0.5
        if valid.sum() < 50:
            continue

        p_raw = all_preds[valid, m] * norm_std[m] + norm_mean[m]
        t_raw = all_targets[valid, m] * norm_std[m] + norm_mean[m]

        # Bin predictions and compute mean actual in each bin
        bin_edges = np.linspace(p_raw.min(), p_raw.max() + 1e-8, n_bins + 1)
        bin_means_pred = []
        bin_means_actual = []
        bin_counts = []

        for b in range(n_bins):
            mask_bin = (p_raw >= bin_edges[b]) & (p_raw < bin_edges[b + 1])
            if mask_bin.sum() > 0:
                bin_means_pred.append(p_raw[mask_bin].mean())
                bin_means_actual.append(t_raw[mask_bin].mean())
                bin_counts.append(mask_bin.sum())

        ax = axes[m]
        bin_means_pred = np.array(bin_means_pred)
        bin_means_actual = np.array(bin_means_actual)
        bin_counts = np.array(bin_counts)

        # Size points by count
        sizes = np.clip(bin_counts / bin_counts.max() * 200, 20, 200)
        ax.scatter(bin_means_pred, bin_means_actual, s=sizes, color=_get_color(marks[m]),
                   alpha=0.7, edgecolor='white', linewidth=1)

        # Perfect calibration line
        lo = min(bin_means_pred.min(), bin_means_actual.min())
        hi = max(bin_means_pred.max(), bin_means_actual.max())
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, linewidth=1.5, label='Perfect')
        _style_ax(ax, f'{marks[m]}', 'Mean Predicted', 'Mean Actual')
        ax.legend(fontsize=8)

    fig.suptitle('Calibration Plot (point size ∝ sample count)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 5. ROC + PR CURVES - per mark classification performance
# ═══════════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════════
# 6. RESIDUAL DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_residual_distributions(all_preds, all_targets, all_masks, output_path):
    marks = config.TARGET_MARKS
    n = len(marks)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for m in range(n):
        valid = all_masks[:, m] > 0.5
        if valid.sum() < 2:
            continue
        res = all_preds[valid, m] - all_targets[valid, m]
        ax = axes[m]
        ax.hist(res, bins=60, color=_get_color(marks[m]), alpha=0.8, edgecolor='white')
        ax.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax.axvline(res.mean(), color='black', linestyle='-', alpha=0.8,
                   label=f'mean={res.mean():.3f}')
        _style_ax(ax, marks[m], 'Residual (pred − true)', '')
        ax.legend(fontsize=8)
    fig.suptitle('Residual Distributions (Z-scored)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 7. SIGNAL DISTRIBUTION OVERLAYS
# ═══════════════════════════════════════════════════════════════════════════

def plot_val_signal_distributions(all_preds, all_targets, all_masks,
                                  norm_mean, norm_std, output_path):
    marks = config.TARGET_MARKS
    n = len(marks)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)
    for m in range(n):
        valid = all_masks[:, m] > 0.5
        if valid.sum() < 2:
            continue
        p_raw = all_preds[valid, m] * norm_std[m] + norm_mean[m]
        t_raw = all_targets[valid, m] * norm_std[m] + norm_mean[m]
        vr = (min(float(t_raw.min()), float(p_raw.min())),
              max(float(t_raw.max()), float(p_raw.max())) + 0.1)

        ax = axes[0, m]
        ax.hist(t_raw, bins=50, range=vr, density=True, color='grey', alpha=0.55, label='Actual')
        ax.hist(p_raw, bins=50, range=vr, density=True, color=_get_color(marks[m]), alpha=0.55, label='Predicted')
        _style_ax(ax, f'{marks[m]} - All', 'log₂(sig+1)', 'Density')
        ax.legend(fontsize=8)

        is_pos = t_raw > 0
        ax = axes[1, m]
        if (~is_pos).sum() > 0:
            ax.hist(p_raw[~is_pos], bins=50, range=vr, density=True,
                    color='#3498db', alpha=0.5, label='Pred (actual=0)')
        if is_pos.sum() > 0:
            ax.hist(p_raw[is_pos], bins=50, range=vr, density=True,
                    color=_get_color(marks[m]), alpha=0.5, label='Pred (actual>0)')
        _style_ax(ax, f'{marks[m]} - Neg vs Pos', 'log₂(sig+1)', 'Density')
        ax.legend(fontsize=8)

    fig.suptitle('Validation Signal Distributions', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT - called from train.py
# ═══════════════════════════════════════════════════════════════════════════

def save_epoch_visualizations(epoch, history, preds, targets, masks, output_dir,
                              norm_mean=None, norm_std=None):
    """Generate all visualizations for one epoch."""
    os.makedirs(output_dir, exist_ok=True)
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    # 1. Master Dashboard (always overwrite global + save per-epoch)
    if norm_mean is not None and norm_std is not None:
        plot_epoch_dashboard(epoch, history, preds, targets, masks,
                             norm_mean, norm_std,
                             os.path.join(output_dir, "dashboard_latest.png"))
        plot_epoch_dashboard(epoch, history, preds, targets, masks,
                             norm_mean, norm_std,
                             os.path.join(epoch_dir, "dashboard.png"))

    # 2. Peak vs Background Violin
    if norm_mean is not None and norm_std is not None:
        plot_peak_vs_background(preds, targets, masks, norm_mean, norm_std,
                                os.path.join(epoch_dir, "peak_vs_background.png"))

    # 3. Mark Correlation Heatmap
    plot_mark_correlation_heatmap(preds, masks,
                                 os.path.join(epoch_dir, "mark_correlation.png"))

    # 4. Calibration Plot
    if norm_mean is not None and norm_std is not None:
        plot_calibration(preds, targets, masks, norm_mean, norm_std,
                         os.path.join(epoch_dir, "calibration.png"))



    # 6. Residual Distributions
    plot_residual_distributions(preds, targets, masks,
                                os.path.join(epoch_dir, "residuals.png"))

    # 7. Signal Distribution Overlays
    if norm_mean is not None and norm_std is not None:
        plot_val_signal_distributions(preds, targets, masks, norm_mean, norm_std,
                                      os.path.join(epoch_dir, "signal_distributions.png"))


# ═══════════════════════════════════════════════════════════════════════════
# HELD-OUT TEST DISTRIBUTIONS (used by test_best_model.py)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_density(values, bins=50, value_range=None):
    if len(values) == 0:
        return None, None
    hist, edges = np.histogram(values, bins=bins, range=value_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist

def plot_heldout_distributions(pred_records, output_dir, bins=50):
    os.makedirs(output_dir, exist_ok=True)
    for mark_name in config.TARGET_MARKS:
        mark_df = pred_records[pred_records["mark"] == mark_name]
        if len(mark_df) == 0:
            continue
        negatives = mark_df[mark_df["present_true"] == 0]
        positives = mark_df[mark_df["present_true"] == 1]
        all_vals = np.concatenate([
            mark_df["actual"].to_numpy(dtype=np.float32),
            mark_df["predicted"].to_numpy(dtype=np.float32),
        ])
        lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
        if lo == hi:
            hi = lo + 1.0
        value_range = (lo, hi)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Held-out: {mark_name}", fontsize=16, fontweight='bold')

        ax = axes[0, 0]
        if len(negatives) > 0:
            ax.hist(negatives["predicted"].to_numpy(dtype=np.float32), bins=bins,
                    range=value_range, density=True, color="#3498db", alpha=0.8, label="Predicted")
        _style_ax(ax, "Predicted for Negatives", "log₂(sig+1)", "Density")
        ax.legend()

        ax = axes[0, 1]
        if len(positives) > 0:
            ax.hist(positives["actual"].to_numpy(dtype=np.float32), bins=bins,
                    range=value_range, density=True, color="grey", alpha=0.55, label="Actual")
            ax.hist(positives["predicted"].to_numpy(dtype=np.float32), bins=bins,
                    range=value_range, density=True, color=_get_color(mark_name), alpha=0.55, label="Predicted")
        _style_ax(ax, "Positive Samples", "log₂(sig+1)", "Density")
        ax.legend()

        ax = axes[1, 0]
        if len(negatives) > 0:
            ax.hist(negatives["predicted"].to_numpy(dtype=np.float32), bins=bins,
                    range=value_range, density=True, color="#3498db", alpha=0.5, label="Pred (neg)")
        if len(positives) > 0:
            ax.hist(positives["predicted"].to_numpy(dtype=np.float32), bins=bins,
                    range=value_range, density=True, color=_get_color(mark_name), alpha=0.5, label="Pred (pos)")
        _style_ax(ax, "Neg vs Pos Predictions", "log₂(sig+1)", "Density")
        ax.legend()

        ax = axes[1, 1]
        ax.hist(mark_df["actual"].to_numpy(dtype=np.float32), bins=bins,
                range=value_range, density=True, color="grey", alpha=0.55, label="Actual")
        ax.hist(mark_df["predicted"].to_numpy(dtype=np.float32), bins=bins,
                range=value_range, density=True, color=_get_color(mark_name), alpha=0.65, label="Predicted")
        _style_ax(ax, "All Held-out", "log₂(sig+1)", "Density")
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heldout_{mark_name}.png"), dpi=200)
        plt.close(fig)
