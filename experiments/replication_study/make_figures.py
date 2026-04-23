"""Generate the three headline figures for Section 5.8 of the report.

Outputs go to results/figures/ at 200 DPI.

Fig 1: fpr_recall_comparison.png
  Bar chart comparing baseline vs augmented detector on 4 categories:
  Set A FPR, Set B recall, Set D FPR, Set E recall.

Fig 2: ablation_curve.png
  Line plot: mean injection probability (B and E) as function of K features
  zeroed, for three rankings: intent-ranked, coef-ranked, random.

Fig 3: feature_activation_comparison.png
  Per-feature bar chart for a selection of key features showing mean
  activation in Set A, B, C, D, E — visualizes intent-discriminators and
  overlap features.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display available
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT.parent.parent / "results" / "figures"
OUT.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def fig1_fpr_recall():
    """Before/after bar chart for the augmentation fix."""
    # Numbers from experiments/replication_study/results/unified_augmentation.json
    # and retrain_comparison.json
    categories = [
        "Set A FPR\n(benign identity)",
        "Set D FPR\n(benign commands)",
        "Set B recall\n(injection identity)",
        "Set E recall\n(injection commands)",
    ]
    baseline = [0.960, 0.640, 1.000, 1.000]   # from replication_results/command_category
    augmented = [0.000, 0.000, 1.000, 1.000]  # from unified_augmentation
    # For baseline injection sets we use the C=1.0 baseline (full-features) numbers
    # which were already 100%; the augmented detector preserves this.

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(categories))
    width = 0.36
    b1 = ax.bar(x - width/2, baseline, width, label="Baseline detector",
                color="#6b7280", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width/2, augmented, width, label="+80 contrastive prompts (augmented)",
                color="#2563EB", edgecolor="black", linewidth=0.6)

    for rect in b1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 0.015, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9)
    for rect in b2:
        h = rect.get_height()
        label = f"{h:.2f}" if h > 0 else "0.00"
        ax.text(rect.get_x() + rect.get_width()/2, h + 0.015, label,
                ha="center", va="bottom", fontsize=9,
                color="#1e40af" if h > 0 else "#dc2626")

    ax.axhspan(0.85, 1.0, alpha=0.07, color="#ef4444")
    ax.text(3.48, 0.92, "blocked (≥0.85)", fontsize=8, color="#991b1b",
            ha="right", va="center", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate at threshold 0.85")
    ax.set_title("Augmentation eliminates both false-positive categories\n"
                 "while preserving injection recall")
    ax.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    out = OUT / "replication_fpr_recall.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


def fig2_ablation_curve():
    """Graduated ablation: mean probability on B/E vs K features zeroed,
    comparing intent-ranked, coef-ranked, and random rankings."""
    with open(ROOT / "results" / "ablation_curve.json") as f:
        data = json.load(f)
    rows = data["rows"]

    Ks = [r["K"] for r in rows]
    B_intent = [r["intent"]["B_mean"] for r in rows]
    B_coef = [r["by_coef"]["B_mean"] for r in rows]
    B_random = [r.get("random_B", r["intent"]["B_mean"] if r["K"] == 0 else None)
                for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(Ks, B_coef, "o-", color="#dc2626", linewidth=2, markersize=7,
            label="Top-K by coefficient magnitude (targeted)")
    ax.plot(Ks, B_intent, "s-", color="#2563EB", linewidth=2, markersize=7,
            label="Top-K intent-ranked")
    ax.plot(Ks, B_random, "D--", color="#6b7280", linewidth=1.6, markersize=6,
            label="Random K features (avg of 3)")

    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Number of SAE features zeroed at test time (K)")
    ax.set_ylabel("Mean injection probability on Set B")
    ax.set_title("Causal ablation: targeted feature removal degrades detection;\n"
                 "random removal does not")
    ax.axhline(0.85, color="#991b1b", linestyle=":", linewidth=1,
               alpha=0.6, label="Blocking threshold (0.85)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", framealpha=0.95)

    # Annotate the key crossover
    ax.annotate("K=320:\ntargeted drops to 0.51;\nrandom stays at 1.00",
                xy=(320, 0.51), xytext=(320, 0.18),
                ha="center", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#444"))

    plt.tight_layout()
    out = OUT / "replication_ablation_curve.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


def fig3_feature_activations():
    """Mean activation across A/B/C/D/E for a curated set of top features
    from each category (intent-discriminators and overlap)."""
    feats_A = np.load(ROOT / "activations" / "A_benign_identity.npy")
    feats_B = np.load(ROOT / "activations" / "B_injection_identity.npy")
    feats_C = np.load(ROOT / "activations" / "C_mundane_control.npy")
    feats_D = np.load(ROOT / "activations" / "D_benign_command.npy")
    feats_E = np.load(ROOT / "activations" / "E_injection_command.npy")

    # Key features identified in the study:
    # - Identity intent-discriminators: 6712, 8319, 8217
    # - Identity overlap: 1680, 6852
    # - Command intent-discriminators: 2324, 2065
    # - Command overlap: 6205, 3702
    # - Cross-domain contributor (attribution): 2594, 6797, 826
    feature_groups = [
        ("Identity intent", [6712, 8319, 8217], "#2563EB"),
        ("Identity topic overlap", [1680, 6852], "#F59E0B"),
        ("Command intent", [2324, 2065], "#16a34a"),
        ("Command topic overlap", [6205, 3702], "#9333ea"),
        ("Cross-domain attribution top", [2594, 6797], "#dc2626"),
    ]

    n_groups = len(feature_groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(16, 4.5), sharey=True)

    sets = [("A", feats_A.mean(0), "#1e40af"),
            ("B", feats_B.mean(0), "#b91c1c"),
            ("C", feats_C.mean(0), "#6b7280"),
            ("D", feats_D.mean(0), "#047857"),
            ("E", feats_E.mean(0), "#c026d6")]

    for i, (group_name, feat_ids, color) in enumerate(feature_groups):
        ax = axes[i]
        x = np.arange(len(feat_ids))
        width = 0.15
        for j, (sname, means, c) in enumerate(sets):
            vals = [means[fi] for fi in feat_ids]
            ax.bar(x + (j - 2) * width, vals, width,
                   label=f"Set {sname}" if i == 0 else None,
                   color=c, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"f{fi}" for fi in feat_ids], fontsize=9)
        ax.set_title(group_name, fontsize=10, color=color, weight="bold")
        ax.grid(axis="y", alpha=0.3)
        if i == 0:
            ax.set_ylabel("Mean activation across set")

    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.95,
                   labelspacing=0.3)
    fig.suptitle("Per-feature mean activation across prompt sets — visualizing "
                 "intent-discriminators, topic overlap, and attribution leaders",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = OUT / "replication_feature_activations.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_fpr_recall()
    fig2_ablation_curve()
    fig3_feature_activations()
    print(f"\nAll figures saved to: {OUT}")
