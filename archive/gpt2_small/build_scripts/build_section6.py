"""Write section 6 of app.py: Red Team Lab, Evasion Lab, System Analysis tabs + launch."""
content = r'''

        # ==============================================================
        # TAB 4: Red Team Lab
        # ==============================================================
        with gr.Tab("Red Team Lab"):
            gr.Markdown(
                "### Penetration Testing Exercise\n"
                "Progress through 5 challenge levels to test the Neural IDS. "
                "Level 1 asks you to craft a detectable injection; levels 2-5 "
                "challenge you to evade detection using increasingly sophisticated "
                "techniques.\n\n"
                "*Analogous to a structured pentest engagement — "
                "systematically probing each defense layer.*"
            )

            current_level = gr.State(0)
            scores = gr.State([False, False, False, False, False])

            level_selector = gr.Radio(
                choices=["Level 1: Direct Injection", "Level 2: Paraphrase Evasion",
                         "Level 3: Encoding Tricks", "Level 4: Mimicry Attack",
                         "Level 5: Free-Form"],
                label="Select Challenge Level",
                value="Level 1: Direct Injection",
            )
            challenge_desc = gr.Markdown(RED_TEAM_CHALLENGES[0]["description"])
            hint_box = gr.Textbox(
                label="Hint",
                value=RED_TEAM_CHALLENGES[0]["hint"],
                interactive=False, lines=1,
            )

            attack_input = gr.Textbox(
                label="Your attack prompt",
                placeholder="Craft your injection attempt here...",
                lines=3,
            )
            submit_attack = gr.Button("Submit Attack", variant="primary", size="lg")

            attack_result_html = gr.HTML(label="Result")
            score_html = gr.HTML(label="Score")

            def on_level_change(level_str):
                idx = int(level_str.split(":")[0].replace("Level ", "")) - 1
                ch = RED_TEAM_CHALLENGES[idx]
                return ch["description"], ch["hint"], idx

            level_selector.change(
                fn=on_level_change,
                inputs=[level_selector],
                outputs=[challenge_desc, hint_box, current_level],
            )

            def on_submit_attack(text, level_idx, current_scores):
                if not text or not text.strip():
                    return "", _render_score(current_scores), current_scores
                result = pipeline.analyze(text)
                if result is None:
                    return "", _render_score(current_scores), current_scores

                ch = RED_TEAM_CHALLENGES[level_idx]
                cond = ch["success_condition"]
                sae_detected = result["sae_pred"] == 1
                tfidf_detected = result["tfidf_pred"] == 1

                success = False
                if cond == "detected":
                    success = sae_detected
                elif cond == "evaded_sae":
                    success = not sae_detected
                elif cond == "evaded_both":
                    success = not sae_detected and not tfidf_detected

                new_scores = list(current_scores)
                if success:
                    new_scores[level_idx] = True

                # Build result HTML
                if success:
                    result_html = (
                        '<div style="background:#F0FDF4;border:2px solid #16A34A;'
                        'border-radius:8px;padding:16px;text-align:center;">'
                        '<div style="font-size:24px;font-weight:bold;color:#16A34A;">'
                        'CHALLENGE PASSED</div>'
                    )
                else:
                    result_html = (
                        '<div style="background:#FEF2F2;border:2px solid #DC2626;'
                        'border-radius:8px;padding:16px;text-align:center;">'
                        '<div style="font-size:24px;font-weight:bold;color:#DC2626;">'
                        'CHALLENGE FAILED</div>'
                    )

                det_status = "Detected" if sae_detected else "Evaded"
                tf_status = "Detected" if tfidf_detected else "Evaded"
                result_html += (
                    f'<div style="margin-top:8px;">'
                    f'Anomaly IDS (SAE): <b>{det_status}</b> ({result["sae_inject_prob"]:.0%})<br>'
                    f'Signature IDS (TF-IDF): <b>{tf_status}</b> ({result["tfidf_inject_prob"]:.0%})'
                    f'</div></div>'
                )

                return result_html, _render_score(new_scores), new_scores

            submit_attack.click(
                fn=on_submit_attack,
                inputs=[attack_input, current_level, scores],
                outputs=[attack_result_html, score_html, scores],
            )

            def _render_score(sc):
                passed = sum(sc)
                cells = ""
                for i, s in enumerate(sc):
                    color = "#16A34A" if s else "#9CA3AF"
                    icon = "&#10003;" if s else str(i + 1)
                    cells += (
                        f'<div style="width:40px;height:40px;border-radius:50%;'
                        f'background:{color};color:white;display:flex;'
                        f'align-items:center;justify-content:center;font-weight:bold;">'
                        f'{icon}</div>'
                    )
                return (
                    f'<div style="display:flex;gap:8px;align-items:center;margin-top:12px;">'
                    f'{cells}'
                    f'<span style="margin-left:12px;font-weight:bold;">'
                    f'{passed}/5 Levels Passed</span></div>'
                )

            # Generate pentest report
            gr.Markdown("---")
            report_btn = gr.Button("Generate Pentest Report")
            report_output = gr.Textbox(label="Pentest Report", lines=15, interactive=False)

            def generate_report(sc):
                passed = sum(sc)
                level_names = [c["name"] for c in RED_TEAM_CHALLENGES]
                report = "# IRIS Neural IDS — Penetration Test Report\n\n"
                report += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                report += f"**Target:** IRIS Neural IDS v1.0\n"
                report += f"**Tester:** Red Team Operator\n\n"
                report += f"## Executive Summary\n\n"
                report += f"Completed {passed}/5 challenge levels against the Neural IDS.\n\n"

                if passed <= 1:
                    risk = "LOW"
                    report += "The IDS demonstrates strong resilience against tested attack vectors.\n"
                elif passed <= 3:
                    risk = "MEDIUM"
                    report += "The IDS shows vulnerability to intermediate evasion techniques.\n"
                else:
                    risk = "HIGH"
                    report += "The IDS is vulnerable to advanced evasion — defense hardening recommended.\n"

                report += f"\n**Overall Risk Rating: {risk}**\n\n"
                report += "## Findings\n\n"
                for i, (name, passed_level) in enumerate(zip(level_names, sc)):
                    status = "PASSED" if passed_level else "FAILED"
                    report += f"- Level {i+1} ({name}): {status}\n"

                report += "\n## Recommendations\n\n"
                report += "1. Train SAE on adversarial examples to improve mimicry detection\n"
                report += "2. Add semantic similarity checks as a third detection layer\n"
                report += "3. Implement rate limiting and input sanitization at the perimeter\n"
                report += "4. Deploy continuous monitoring via the Neural IDS Console\n"
                return report

            report_btn.click(fn=generate_report, inputs=[scores], outputs=[report_output])

        # ==============================================================
        # TAB 5: Evasion Lab
        # ==============================================================
        with gr.Tab("Evasion Lab"):
            gr.Markdown(
                "### Adversarial Evasion Experimentation\n"
                "Compare an original injection against a modified version. "
                "See how signature activations change when you modify the attack.\n\n"
                "*This is a live version of experiment C4 (adversarial evasion). "
                "Analogous to testing IDS bypass techniques in a security lab.*"
            )

            with gr.Row():
                original_input = gr.Textbox(
                    label="Original Injection",
                    value=EXAMPLES_EVASION_ORIGINAL,
                    lines=3,
                )
                modified_input = gr.Textbox(
                    label="Evasion Attempt (modify this)",
                    value=EXAMPLES_EVASION_MODIFIED[0],
                    lines=3,
                )

            compare_btn = gr.Button("Compare", variant="primary", size="lg")

            gr.Examples(
                examples=[
                    [EXAMPLES_EVASION_ORIGINAL, e] for e in EXAMPLES_EVASION_MODIFIED
                ],
                inputs=[original_input, modified_input],
                label="Example evasion strategies",
            )

            evasion_plot = gr.Plot(label="Signature Activation Comparison")

            with gr.Row():
                evasion_orig_html = gr.HTML(label="Original Result")
                evasion_mod_html = gr.HTML(label="Modified Result")

            def on_compare(orig, mod):
                if not orig or not mod or not orig.strip() or not mod.strip():
                    return None, "", ""
                r1 = pipeline.analyze(orig)
                r2 = pipeline.analyze(mod)
                if r1 is None or r2 is None:
                    return None, "", ""
                fig = _evasion_comparison_plot(r1, r2)

                evaded = r1["sae_pred"] == 1 and r2["sae_pred"] == 0
                if evaded:
                    mod_summary = (
                        '<div style="background:#FEF2F2;border:2px solid #DC2626;'
                        'border-radius:8px;padding:16px;text-align:center;">'
                        '<b style="color:#DC2626;font-size:18px;">EVASION SUCCESSFUL</b>'
                        '<br>The modified prompt evaded the anomaly-based detector.'
                        f'<br>Threat probability: {r1["sae_inject_prob"]:.0%} '
                        f'&rarr; {r2["sae_inject_prob"]:.0%}</div>'
                    )
                else:
                    mod_summary = (
                        '<div style="background:#F0FDF4;border:2px solid #16A34A;'
                        'border-radius:8px;padding:16px;text-align:center;">'
                        '<b style="color:#16A34A;font-size:18px;">EVASION BLOCKED</b>'
                        '<br>The Neural IDS still detected the modified prompt.'
                        f'<br>Threat probability: {r2["sae_inject_prob"]:.0%}</div>'
                    )

                orig_summary = (
                    '<div style="background:#F9FAFB;border:1px solid #E5E7EB;'
                    'border-radius:8px;padding:16px;text-align:center;">'
                    f'<b>Original:</b> {"ALERT" if r1["sae_pred"]==1 else "PASS"} '
                    f'({r1["sae_inject_prob"]:.0%})</div>'
                )

                return fig, orig_summary, mod_summary

            compare_btn.click(
                fn=on_compare,
                inputs=[original_input, modified_input],
                outputs=[evasion_plot, evasion_orig_html, evasion_mod_html],
            )

        # ==============================================================
        # TAB 6: System Analysis
        # ==============================================================
        with gr.Tab("System Analysis"):
            gr.Markdown(
                "### Security Analysis & Key Results\n"
                "STRIDE threat model, kill chain decomposition, defense-in-depth "
                "architecture, and the mapping between IRIS components and "
                "traditional network security tools."
            )
            gr.HTML(_system_analysis_html(pipeline))

    return app


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------

def launch(project_root=".", share=True, **kwargs):
    """Load the pipeline and launch the Gradio app.

    Args:
        project_root: Path to the IRIS project root.
        share: Create a public Gradio URL (True for Colab).
        **kwargs: Additional kwargs passed to app.launch().
    """
    pipeline = IRISPipeline(project_root)
    pipeline.load()
    app = build_app(pipeline)
    app.launch(share=share, **kwargs)


if __name__ == "__main__":
    launch(share=False)
'''

with open('src/app.py', 'a', encoding='utf-8') as f:
    f.write(content)
print(f'Section 6: {len(content.splitlines())} lines')
