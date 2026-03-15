"""Write section 4 of app.py: System Analysis HTML."""
content = '''

# ---------------------------------------------------------------------------
# Tab content: System Analysis HTML
# ---------------------------------------------------------------------------

def _system_analysis_html(pipeline):
    """Build the static system analysis tab content."""
    c3 = pipeline.results.get("c3_detection_comparison", {})
    c3_rows = ""
    if "results" in c3:
        for name, m in c3["results"].items():
            short = name.replace(" + LogReg", "").replace(
                " + Logistic Regression", " + LR"
            )
            c3_rows += (
                '<tr><td style="padding:6px 12px;">' + short + '</td>'
                '<td style="padding:6px 12px;text-align:center;">'
                + f'{m["f1"]:.3f}</td>'
                '<td style="padding:6px 12px;text-align:center;">'
                + f'{m["roc_auc"]:.3f}</td></tr>'
            )

    c4 = pipeline.results.get("c4_adversarial_evasion", {})
    c4_rows = ""
    if "per_strategy" in c4:
        for strategy, data in c4["per_strategy"].items():
            rate = data["evasion_rate"]
            color = "#DC2626" if rate > 0.5 else "#F59E0B" if rate > 0 else "#16A34A"
            c4_rows += (
                '<tr><td style="padding:6px 12px;text-transform:capitalize;">'
                + strategy + '</td>'
                '<td style="padding:6px 12px;text-align:center;color:' + color
                + ';font-weight:bold;">' + f'{rate:.0%}</td>'
                '<td style="padding:6px 12px;text-align:center;">'
                + f'{data["evaded"]}/{data["total"]}</td></tr>'
            )

    j1 = pipeline.results.get("j1_separability", {})
    j1_sil = j1.get("0", {}).get("silhouette", 0)
    j1_d = j1.get("0", {}).get("cohens_d", 0)
    j1_pass = j1.get("j1_passed", False)
    overall_ev = c4.get("overall_evasion_rate", 0)

    pass_text = "PASS" if j1_pass else "FAIL"
    ev_bg = "#FEF2F2" if overall_ev > 0.3 else "#FFFBEB"
    ev_border = "#FECACA" if overall_ev > 0.3 else "#FDE68A"
    ev_color = "#DC2626" if overall_ev > 0.3 else "#F59E0B"

    html = (
        '<div style="max-width:900px;margin:0 auto;">'

        # -- Concept mapping table --
        '<h3>Network Security Concept Mapping</h3>'
        '<p style="color:#666;font-size:13px;">IRIS applies network security '
        'principles to LLM internals. Each IRIS component maps to a familiar '
        'network security tool.</p>'
        '<table style="width:100%;border-collapse:collapse;background:white;'
        'border:1px solid #E5E7EB;">'
        '<tr style="background:#1E3A5F;color:white;">'
        '<th style="padding:8px 12px;text-align:left;">IRIS Component</th>'
        '<th style="padding:8px 12px;text-align:left;">Network Security Analogue</th>'
        '<th style="padding:8px 12px;text-align:left;">Function</th></tr>'
    )
    concept_rows = [
        ("SAE feature activations", "Packet payload inspection", "Deep content analysis"),
        ("Sensitivity scores", "IDS signature rules (Snort SIDs)", "Pattern matching confidence"),
        ("Feature thresholds", "Firewall allow/deny rules", "Binary pass/block decision"),
        ("TF-IDF detector", "Signature-based IDS (Snort)", "Known-pattern matching"),
        ("SAE detector", "Anomaly-based IDS (behavioral)", "Deviation from baseline"),
        ("Dual-detector consensus", "Defense-in-depth / layered security", "Multiple detection layers"),
        ("Evasion Lab", "Penetration testing", "Adversarial robustness testing"),
        ("Mimicry evasion", "Zero-day exploit", "No existing signature matches"),
        ("Top-K feature selection", "Ruleset tuning", "Reduce alert fatigue"),
        ("IDS Console log", "SIEM event log (Splunk/ELK)", "Centralized alert monitoring"),
    ]
    for i, (iris, net, func) in enumerate(concept_rows):
        bg = ' style="background:#F9FAFB;"' if i % 2 == 1 else ""
        html += (
            f'<tr{bg}><td style="padding:6px 12px;">{iris}</td>'
            f'<td style="padding:6px 12px;">{net}</td>'
            f'<td style="padding:6px 12px;">{func}</td></tr>'
        )
    html += '</table>'

    # -- Metric cards --
    html += (
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0;">'
        '<div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;'
        'padding:16px;text-align:center;">'
        '<div style="font-size:12px;color:#666;text-transform:uppercase;">J1 Separability</div>'
        f'<div style="font-size:24px;font-weight:bold;color:#16A34A;">{pass_text}</div>'
        f'<div style="font-size:11px;color:#888;">d={j1_d:.1f}, sil={j1_sil:.3f}</div></div>'
        '<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;'
        'padding:16px;text-align:center;">'
        '<div style="font-size:12px;color:#666;text-transform:uppercase;">Signatures</div>'
        '<div style="font-size:24px;font-weight:bold;color:#2563EB;">6,144</div>'
        '<div style="font-size:11px;color:#888;">8x expansion SAE</div></div>'
        '<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;'
        'padding:16px;text-align:center;">'
        '<div style="font-size:12px;color:#666;text-transform:uppercase;">SAE F1</div>'
        '<div style="font-size:24px;font-weight:bold;color:#2563EB;">0.946</div>'
        '<div style="font-size:11px;color:#888;">vs Snort-style: 0.956</div></div>'
        f'<div style="background:{ev_bg};border:1px solid {ev_border};border-radius:8px;'
        f'padding:16px;text-align:center;">'
        '<div style="font-size:12px;color:#666;text-transform:uppercase;">Evasion Rate</div>'
        f'<div style="font-size:24px;font-weight:bold;color:{ev_color};">{overall_ev:.0%}</div>'
        f'<div style="font-size:11px;color:#888;">'
        f'{c4.get("evaded",0)}/{c4.get("n_evasion_prompts",0)} evaded</div></div></div>'
    )

    # -- C3 + C4 tables --
    html += (
        '<h3>Detection Pipeline Comparison (C3)</h3>'
        '<table style="width:100%;border-collapse:collapse;background:white;'
        'border:1px solid #E5E7EB;">'
        '<tr style="background:#F3F4F6;">'
        '<th style="padding:8px 12px;text-align:left;">Approach</th>'
        '<th style="padding:8px 12px;text-align:center;">F1</th>'
        '<th style="padding:8px 12px;text-align:center;">AUC</th></tr>'
        + c3_rows + '</table>'
        '<h3 style="margin-top:24px;">Adversarial Evasion Results (C4)</h3>'
        '<table style="width:100%;border-collapse:collapse;background:white;'
        'border:1px solid #E5E7EB;">'
        '<tr style="background:#F3F4F6;">'
        '<th style="padding:8px 12px;text-align:left;">Strategy</th>'
        '<th style="padding:8px 12px;text-align:center;">Evasion Rate</th>'
        '<th style="padding:8px 12px;text-align:center;">Evaded / Total</th></tr>'
        + c4_rows + '</table>'
    )

    # -- STRIDE --
    html += (
        '<h3 style="margin-top:24px;">STRIDE Threat Model</h3>'
        '<table style="width:100%;border-collapse:collapse;background:white;'
        'border:1px solid #E5E7EB;">'
        '<tr style="background:#F3F4F6;">'
        '<th style="padding:8px 12px;text-align:left;">Category</th>'
        '<th style="padding:8px 12px;text-align:left;">Key Threat</th>'
        '<th style="padding:8px 12px;text-align:center;">Risk</th></tr>'
    )
    stride_rows = [
        ("Spoofing", "Attacker impersonates system prompt", "High", "#DC2626"),
        ("Tampering", "Injection modifies model behavior", "Critical", "#DC2626"),
        ("Repudiation", "No audit trail for injected actions", "Medium", "#F59E0B"),
        ("Info Disclosure", "System prompt extraction via injection", "High", "#DC2626"),
        ("Denial of Service", "Resource exhaustion via crafted prompts", "Medium", "#F59E0B"),
        ("Elevation of Privilege", "Injection grants tool/API access", "Critical", "#DC2626"),
    ]
    for cat, threat, risk, rcolor in stride_rows:
        html += (
            f'<tr><td style="padding:6px 12px;">{cat}</td>'
            f'<td style="padding:6px 12px;">{threat}</td>'
            f'<td style="padding:6px 12px;text-align:center;color:{rcolor};'
            f'font-weight:bold;">{risk}</td></tr>'
        )
    html += '</table>'

    # -- Kill chain --
    html += '<h3 style="margin-top:24px;">Kill Chain: Prompt Injection Attack</h3>'
    html += '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;">'
    kc_steps = [
        ("1. Recon", "Probe system prompt", "#FEF2F2", "#FECACA"),
        ("2. Weaponize", "Craft payload", "#FFF7ED", "#FED7AA"),
        ("3. Deliver", "Submit via input", "#FFFBEB", "#FDE68A"),
        ("4. Exploit", "Model follows injection", "#FEF9C3", "#FDE047"),
        ("5. Impact", "Data exfil / escalation", "#ECFDF5", "#A7F3D0"),
    ]
    for title, desc, bg, border in kc_steps:
        html += (
            f'<div style="flex:1;min-width:150px;background:{bg};border:1px solid {border};'
            f'border-radius:8px;padding:12px;text-align:center;">'
            f'<div style="font-weight:bold;">{title}</div>'
            f'<div style="font-size:11px;color:#666;">{desc}</div></div>'
        )
    html += '</div>'

    # -- Defense in depth --
    html += (
        '<h3 style="margin-top:24px;">Defense-in-Depth Architecture</h3>'
        '<table style="width:100%;border-collapse:collapse;background:white;'
        'border:1px solid #E5E7EB;">'
        '<tr style="background:#F3F4F6;">'
        '<th style="padding:8px 12px;text-align:left;">Layer</th>'
        '<th style="padding:8px 12px;text-align:left;">Defense</th>'
        '<th style="padding:8px 12px;text-align:left;">Network Analogue</th></tr>'
    )
    did_rows = [
        ("1. Input Filtering", "TF-IDF pattern matching", "Perimeter firewall"),
        ("2. Deep Inspection", "SAE feature decomposition", "Deep packet inspection"),
        ("3. Anomaly Detection", "Neural feature classifier", "Behavioral IDS"),
        ("4. Alert & Response", "Verdict + logging", "SIEM + SOAR"),
        ("5. Audit", "Feature activation logs", "Forensic packet capture"),
    ]
    for layer, defense, analogue in did_rows:
        html += (
            f'<tr><td style="padding:6px 12px;">{layer}</td>'
            f'<td style="padding:6px 12px;">{defense}</td>'
            f'<td style="padding:6px 12px;">{analogue}</td></tr>'
        )
    html += '</table></div>'

    return html
'''

with open('src/app.py', 'a', encoding='utf-8') as f:
    f.write(content)
print(f'Section 4: {len(content.splitlines())} lines')
