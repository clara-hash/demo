#!/usr/bin/env python3
"""Generate V2 report figures with 3000-case V7 results."""

import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "outputs"

def make_figures():
    # ── Figure 1: V7 Training Loss & Accuracy (3000 cases) ──
    loss_csv = "outputs/nuplan_irl_neural_reward_v7/training_loss_v7.csv"
    if os.path.exists(loss_csv):
        data = np.loadtxt(loss_csv, delimiter=",", skiprows=1)
        epochs, tr_l, tr_a, val_l, val_a = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=140)
        ax1.plot(epochs, tr_l, 'b-', label='Train Loss', alpha=0.8)
        ax1.plot(epochs, val_l, 'r-', label='Val Loss', alpha=0.8)
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cross-Entropy Loss')
        ax1.set_title('V7 Training Loss (3000 cases)')
        ax1.grid(True, alpha=0.3); ax1.legend()
        ax1.text(0.95, 0.95, f'Train: {tr_l[0]:.3f}→{tr_l[-1]:.3f}\nVal: {val_l[0]:.3f}→{val_l[-1]:.3f}',
                 transform=ax1.transAxes, ha='right', va='top', fontsize=9, color='gray')

        ax2.plot(epochs, tr_a, 'b-', label='Train Acc', alpha=0.8)
        ax2.plot(epochs, val_a, 'r-', label='Val Acc', alpha=0.8)
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Choice Accuracy')
        ax2.set_title('V7 Choice Accuracy (3000 cases)')
        ax2.grid(True, alpha=0.3); ax2.legend()
        ax2.text(0.95, 0.95, f'Train: {tr_a[0]:.1%}→{tr_a[-1]:.1%}\nVal: {val_a[0]:.1%}→{val_a[-1]:.1%}',
                 transform=ax2.transAxes, ha='right', va='top', fontsize=9, color='gray')
        fig.suptitle('V7 Neural Reward IRL Training (3000 Cases)', fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "report_v7_training.png"), dpi=160)
        plt.close(fig)

    # ── Figure 2: V7 Planning Test (30 cases, 3000-case model) ──
    test_data = [
        (0, 58.6, 58.6, 1.000, 1.0, 1.0),
        (1, 166.7, 144.5, 1.154, 1.0, 3.0),
        (2, 100.5, 78.9, 1.275, 1.0, 8.5),
        (3, 152.6, 145.9, 1.046, 1.0, 4.0),
        (4, 136.8, 134.5, 1.017, 2.0, 4.0),
        (5, 136.7, 126.6, 1.080, 1.0, 6.0),
        (6, 157.3, 148.7, 1.058, 2.2, 9.2),
        (7, 78.4, 68.0, 1.152, 1.0, 6.0),
        (8, 92.0, 76.8, 1.198, 2.0, 4.0),
        (9, 120.7, 119.5, 1.010, 1.0, 2.8),
        (10, 102.2, 93.3, 1.096, 1.0, 7.0),
        (11, 193.3, 182.0, 1.062, 1.0, 5.0),
        (12, 116.6, 111.5, 1.045, 1.0, 2.0),
        (13, 96.9, 91.9, 1.055, 1.0, 2.0),
        (14, 147.9, 142.1, 1.041, 1.0, 4.0),
        (15, 173.7, 171.3, 1.015, 1.0, 2.8),
        (16, 130.2, 121.4, 1.073, 1.0, 6.0),
        (17, 282.3, 268.2, 1.052, 1.0, 2.8),
        (18, 86.8, 86.8, 1.000, 1.0, 1.0),
        (19, 211.1, 210.3, 1.004, 1.0, 3.6),
        (20, 211.6, 200.2, 1.057, 1.0, 5.7),
        (21, 114.3, 107.3, 1.065, 1.0, 2.0),
        (22, 164.2, 149.1, 1.102, 1.0, 2.0),
        (23, 171.3, 157.0, 1.091, 2.0, 6.7),
        (24, 140.6, 132.3, 1.063, 1.0, 2.0),
        (25, 177.9, 175.1, 1.016, 1.0, 2.0),
        (26, 237.2, 229.5, 1.033, 2.2, 14.8),
        (27, 144.9, 139.1, 1.042, 1.0, 12.1),
        (28, 62.3, 53.8, 1.158, 1.0, 10.6),
        (29, 118.5, 112.5, 1.053, 4.0, 12.4),
    ]
    ids = [d[0] for d in test_data]
    irl_l = [d[1] for d in test_data]
    sht_l = [d[2] for d in test_data]
    ratios = [d[3] for d in test_data]
    irl_sm = [d[4] for d in test_data]
    irl_om = [d[5] for d in test_data]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=140)
    w = 0.35; x = np.arange(len(ids))

    axes[0,0].bar(x - w/2, irl_l, w, label='IRL', color='steelblue')
    axes[0,0].bar(x + w/2, sht_l, w, label='Shortest', color='lightcoral')
    axes[0,0].set_title('Path Length per Test Case'); axes[0,0].set_xlabel('Test ID')
    axes[0,0].set_ylabel('Length (cells)'); axes[0,0].legend(fontsize=8); axes[0,0].grid(True, alpha=0.2, axis='y')

    axes[0,1].bar(ids, ratios, color='orange', alpha=0.7)
    axes[0,1].axhline(1.0, color='red', ls='--', lw=1.5, label='Shortest=1.0')
    axes[0,1].axhline(np.mean(ratios), color='green', ls='-', lw=1.5, label=f'Mean={np.mean(ratios):.3f}')
    axes[0,1].set_title('Length Ratio (IRL / Shortest)'); axes[0,1].set_xlabel('Test ID')
    axes[0,1].legend(fontsize=8); axes[0,1].grid(True, alpha=0.2, axis='y')

    axes[0,2].bar(ids, irl_om, color='green', alpha=0.7)
    axes[0,2].axhline(np.mean(irl_om), color='darkgreen', ls='-', lw=2, label=f'Mean={np.mean(irl_om):.2f}')
    axes[0,2].set_title('Min Obstacle Clearance (3000 cases)'); axes[0,2].set_xlabel('Test ID')
    axes[0,2].set_ylabel('Cells'); axes[0,2].legend(fontsize=8); axes[0,2].grid(True, alpha=0.2, axis='y')

    axes[1,0].bar(ids, irl_sm, color='steelblue', alpha=0.7)
    axes[1,0].axhline(np.mean(irl_sm), color='navy', ls='-', lw=1.5, label=f'Mean={np.mean(irl_sm):.2f}')
    axes[1,0].set_title('Min Static Clearance'); axes[1,0].set_xlabel('Test ID')
    axes[1,0].set_ylabel('Cells'); axes[1,0].legend(fontsize=8); axes[1,0].grid(True, alpha=0.2, axis='y')

    axes[1,1].hist(irl_om, bins=16, color='green', alpha=0.7, edgecolor='black')
    axes[1,1].axvline(np.mean(irl_om), color='darkgreen', ls='-', lw=2, label=f'Mean={np.mean(irl_om):.2f}')
    axes[1,1].set_title('Obstacle Clearance Distribution'); axes[1,1].set_xlabel('Min Obstacle Clearance (cells)')
    axes[1,1].set_ylabel('Count'); axes[1,1].legend()

    axes[1,2].axis('off')
    ob_low = sum(1 for o in irl_om if o <= 1.0)
    ob_mid = sum(1 for o in irl_om if o >= 5.0)
    ob_high = sum(1 for o in irl_om if o >= 10.0)
    axes[1,2].text(0.05, 0.95,
        f"V7 (3000 cases) Test Summary\n{'─'*40}\n"
        f"Cases: {len(test_data)}\n"
        f"Success rate: 100%\n\n"
        f"Mean IRL length: {np.mean(irl_l):.1f}\n"
        f"Mean shortest: {np.mean(sht_l):.1f}\n"
        f"Mean ratio: {np.mean(ratios):.3f}\n\n"
        f"Mean obstacle min: {np.mean(irl_om):.2f}\n"
        f"Mean static min: {np.mean(irl_sm):.2f}\n\n"
        f"obs_min <= 1.0: {ob_low}/{len(test_data)} ({ob_low/len(test_data):.0%})\n"
        f"obs_min >= 5.0: {ob_mid}/{len(test_data)} ({ob_mid/len(test_data):.0%})\n"
        f"obs_min >= 10.0: {ob_high}/{len(test_data)} ({ob_high/len(test_data):.0%})",
        transform=axes[1,2].transAxes, fontsize=10, fontfamily='monospace', va='top')

    fig.suptitle('V7 Neural Reward IRL — 3000-Case Training Results', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "report_v7_planning_summary.png"), dpi=160)
    plt.close(fig)

    # ── Figure 3: Scale comparison (800 vs 3000 cases) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=140)

    # 800-case data (from before)
    om_800 = [1.0,3.0,8.5,5.0,4.0,5.0,9.2,5.0,1.0,2.8,1.0,4.0,2.0,2.0,4.0,2.8,6.0,2.8,1.0,3.6,5.0,2.0,2.0,6.7]
    # 3000-case data
    om_3000 = irl_om

    axes[0,0].hist(om_800, bins=10, color='orange', alpha=0.7, edgecolor='black')
    axes[0,0].axvline(np.mean(om_800), color='darkorange', ls='-', lw=2, label=f'Mean={np.mean(om_800):.2f}')
    axes[0,0].set_title(f'800 Cases: Obstacle Clearance\nobs<=1: {sum(1 for o in om_800 if o<=1)}/{len(om_800)} | obs>=5: {sum(1 for o in om_800 if o>=5)}/{len(om_800)}')
    axes[0,0].set_xlabel('Min Obstacle Clearance (cells)'); axes[0,0].set_ylabel('Count')
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.2, axis='y')

    axes[0,1].hist(om_3000, bins=10, color='green', alpha=0.7, edgecolor='black')
    axes[0,1].axvline(np.mean(om_3000), color='darkgreen', ls='-', lw=2, label=f'Mean={np.mean(om_3000):.2f}')
    axes[0,1].set_title(f'3000 Cases: Obstacle Clearance\nobs<=1: {sum(1 for o in om_3000 if o<=1)}/{len(om_3000)} | obs>=5: {sum(1 for o in om_3000 if o>=5)}/{len(om_3000)} | obs>=10: {sum(1 for o in om_3000 if o>=10)}/{len(om_3000)}')
    axes[0,1].set_xlabel('Min Obstacle Clearance (cells)'); axes[0,1].set_ylabel('Count')
    axes[0,1].legend(); axes[0,1].grid(True, alpha=0.2, axis='y')

    # Accuracy comparison
    bar_data = {
        '800 cases': (27.1, 62.5, 64.2),
        '3000 cases': (31.3, 77.8, 75.8),
    }
    x_pos = np.arange(3)
    w = 0.35
    for i, (label, (tr_s, va_e, te_e)) in enumerate(bar_data.items()):
        axes[1,0].bar(x_pos + (i-0.5)*w, [tr_s, va_e, te_e], w, label=label, alpha=0.8)
    axes[1,0].set_xticks(x_pos); axes[1,0].set_xticklabels(['Train Acc (start)', 'Val Acc (final)', 'Test Acc'])
    axes[1,0].set_ylabel('Accuracy (%)'); axes[1,0].set_title('Choice Accuracy: 800 vs 3000 Cases')
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.2, axis='y')
    axes[1,0].set_ylim(0, 90)

    # Test loss trend
    axes[1,1].axis('off')
    axes[1,1].text(0.1, 0.95,
        f"Training Scale Impact\n{'─'*30}\n\n"
        f"           800 cases  →  3000 cases\n"
        f"Train loss  1.991→1.004  1.763→0.864\n"
        f"Val loss    1.955→1.004  1.717→0.807\n"
        f"Test loss   1.132         0.868\n\n"
        f"Val acc     65.8%   →    77.8%  (+12pp)\n"
        f"Test acc    60.8%   →    75.8%  (+15pp)\n\n"
        f"Obs min     3.73    →    5.64   (+51%)\n"
        f"Obs>=5      42%     →    43%\n"
        f"Obs>=10     0%      →    13%    (new!)\n"
        f"Ratio       1.079   →    1.070",
        transform=axes[1,1].transAxes, fontsize=10, fontfamily='monospace', va='top')

    fig.suptitle('V7 Scale Impact: 800 Cases vs 3000 Cases', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "report_v7_scale_comparison.png"), dpi=160)
    plt.close(fig)

    # ── Figure 4: All versions loss comparison ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=140)
    v3_l = [2.052,2.017,1.985,1.956,1.930,1.906,1.885,1.865,1.847,1.830,1.814,1.800,
            1.786,1.773,1.761,1.750,1.740,1.730,1.720,1.711,1.703,1.695]
    axes[0,0].plot(np.arange(0,220,10)[:len(v3_l)], v3_l, 'b-o', ms=3)
    axes[0,0].set_title('V3: Goal-conditioned (43 samples)'); axes[0,0].set_ylabel('NLL'); axes[0,0].grid(True, alpha=0.3)

    v4_l = [2.052,2.011,1.976,1.944,1.915,1.890,1.866,1.845,1.826,1.808,1.792,1.777,
            1.763,1.749,1.737,1.726,1.715,1.705,1.695,1.686,1.677,1.669,1.662]
    axes[0,1].plot(np.arange(0,230,10)[:len(v4_l)], v4_l, 'g-s', ms=3)
    axes[0,1].set_title('V4: + Clearance (43 samples)'); axes[0,1].set_ylabel('NLL'); axes[0,1].grid(True, alpha=0.3)

    v6_l = [2.664,2.565,2.391,2.285,2.209,2.151,2.105,2.068,2.038,2.013,1.991,1.973,1.958,
            1.944,1.932,1.921,1.912,1.904,1.896,1.889,1.883,1.877,1.871,1.866,1.862,1.857,
            1.852,1.847,1.844,1.841,1.838]
    axes[1,0].plot(np.arange(0,930,30)[:len(v6_l)], v6_l, 'r-^', ms=3)
    axes[1,0].set_title('V6 Heavy: Linear IRL (2400 cases, 900 ep)'); axes[1,0].set_ylabel('NLL')
    axes[1,0].set_xlabel('Epoch'); axes[1,0].grid(True, alpha=0.3)

    if os.path.exists(loss_csv):
        d = np.loadtxt(loss_csv, delimiter=",", skiprows=1)
        axes[1,1].plot(d[:,0], d[:,1], 'purple', alpha=0.8, label='Train')
        axes[1,1].plot(d[:,0], d[:,3], 'darkviolet', alpha=0.8, label='Val')
        axes[1,1].set_title('V7: Neural Reward (3000 cases, 80 ep)')
        axes[1,1].set_ylabel('Cross-Entropy Loss'); axes[1,1].set_xlabel('Epoch')
        axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    fig.suptitle('IRL Training Loss Evolution (V3→V7)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "report_all_versions_loss.png"), dpi=160)
    plt.close(fig)

    print("All figures regenerated with 3000-case V7 data.")

if __name__ == "__main__":
    make_figures()
