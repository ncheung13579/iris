content = r'''

# ---------------------------------------------------------------------------
# UI helpers — verdict and comparison HTML
# ---------------------------------------------------------------------------

def _verdict_html(result):
    """Build the verdict banner HTML (SOC alert style)."""
    pred = result["sae_pred"]
    prob = result["sae_inject_prob"]

    if pred == 1:
        color = "#DC2626" if prob > 0.8 else "#F59E0B"
        icon = "&#9888;&#65039;"
        label = "ALERT: INJECTION DETECTED"
        bg = "#FEF2F2" if prob > 0.8 else "#FFFBEB"
    else:
        color = "#16A34A"
        icon = "&#9989;"
        label = "PASS: TRAFFIC CLEAR"
        bg = "#F0FDF4"

    if prob > 0.8:
        threat, tc = "CRITICAL", "#DC2626"
    elif prob > 0.5:
        threat, tc = "WARNING", "#F59E0B"
    else:
        threat, tc = "INFO", "#16A34A"

    return (
        f'<div style="background:{bg};border:3px solid {color};border-radius:12px;'
        f'padding:24px;text-align:center;margin-bottom:16px;">'
        f'<div style="font-size:48px;">{icon}</div>'
        f'<div style="font-size:28px;font-weight:bold;color:{color};">{label}</div>'
        f'<div style="font-size:18px;color:#374151;margin-top:8px;">'
        f'Threat Probability: <b>{prob:.1%}</b></div>'
        f'<div style="margin-top:12px;display:inline-block;padding:4px 16px;'
        f'border-radius:20px;background:{tc};color:white;'
        f'font-weight:bold;font-size:14px;">Severity: {threat}</div></div>'
    )


def _detector_comparison_html(result):
    """Build the dual-detector comparison card (defense-in-depth view)."""
    sae_p = result["sae_pred"]
    tf_p = result["tfidf_pred"]
    sp = result["sae_inject_prob"]
    tp = result["tfidf_inject_prob"]

    sl = "ALERT" if sae_p == 1 else "PASS"
    tl = "ALERT" if tf_p == 1 else "PASS"
    sc = "#DC2626" if sae_p == 1 else "#16A34A"
    tc = "#DC2626" if tf_p == 1 else "#16A34A"

    agree = sae_p == tf_p
    at = "Detectors AGREE" if agree else "Detectors DISAGREE"
    ac = "#16A34A" if agree else "#F59E0B"

    return (
        f'<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;'
        f'padding:16px;margin-top:8px;">'
        f'<div style="font-weight:bold;font-size:16px;margin-bottom:12px;'
        f'color:{ac};">{at} (Defense-in-Depth)</div>'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<tr style="border-bottom:1px solid #E5E7EB;">'
        f'<td style="padding:8px;font-weight:bold;">Detection Layer</td>'
        f'<td style="padding:8px;font-weight:bold;">Type</td>'
        f'<td style="padding:8px;font-weight:bold;">Verdict</td>'
        f'<td style="padding:8px;font-weight:bold;">Threat Prob.</td></tr>'
        f'<tr style="border-bottom:1px solid #E5E7EB;">'
        f'<td style="padding:8px;">Layer 1: Anomaly-based</td>'
        f'<td style="padding:8px;font-size:12px;color:#666;">SAE neural features</td>'
        f'<td style="padding:8px;color:{sc};font-weight:bold;">{sl}</td>'
        f'<td style="padding:8px;">{sp:.1%}</td></tr>'
        f'<tr><td style="padding:8px;">Layer 2: Signature-based</td>'
        f'<td style="padding:8px;font-size:12px;color:#666;">TF-IDF text patterns</td>'
        f'<td style="padding:8px;color:{tc};font-weight:bold;">{tl}</td>'
        f'<td style="padding:8px;">{tp:.1%}</td></tr>'
        f'</table></div>'
    )


def _feature_plot(result):
    """Horizontal bar chart of top 10 signatures with signal strength."""
    feats = result["features"][:10]
    fig, ax = plt.subplots(figsize=(8, 5))

    indices = [f["index"] for f in feats]
    activations = [f["activation"] for f in feats]
    sensitivities = [f["sensitivity"] for f in feats]
    colors = ["#DC2626" if s > 0 else "#2563EB" for s in sensitivities]

    y_labels = [
        f'SID-{idx} ({"inj" if s > 0 else "nor"})'
        for idx, s in zip(indices, sensitivities)
    ]
    ax.barh(y_labels, activations, color=colors, alpha=0.8, edgecolor="white")
    ax.set_xlabel("Signal Strength (activation magnitude)")
    ax.set_title("Top 10 IDS Signatures — Signal Strength")
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color="#DC2626", label="Injection signature"),
            Patch(color="#2563EB", label="Normal-traffic signature"),
        ],
        loc="lower right",
        fontsize=9,
    )
    plt.tight_layout()
    return fig


def _evasion_comparison_plot(r1, r2):
    """Side-by-side signature activation comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, result, title in [
        (axes[0], r1, "Original Injection"),
        (axes[1], r2, "Evasion Attempt"),
    ]:
        feats = result["features"][:10]
        indices = [f["index"] for f in feats]
        activations = [f["activation"] for f in feats]
        sensitivities = [f["sensitivity"] for f in feats]
        colors = ["#DC2626" if s > 0 else "#2563EB" for s in sensitivities]
        y_labels = [f"SID-{idx}" for idx in indices]

        ax.barh(y_labels, activations, color=colors, alpha=0.8, edgecolor="white")
        ax.set_xlabel("Signal Strength")

        pred = result["sae_pred"]
        prob = result["sae_inject_prob"]
        verdict = "ALERT" if pred == 1 else "PASS"
        v_color = "#DC2626" if pred == 1 else "#16A34A"
        ax.set_title(
            f"{title}\n{verdict} ({prob:.0%})",
            color=v_color,
            fontweight="bold",
        )

    axes[0].invert_yaxis()
    plt.tight_layout()
    return fig
'''

with open('src/app.py', 'a', encoding='utf-8') as f:
    f.write(content)
print(f'Section 3: {len(content.splitlines())} lines')
