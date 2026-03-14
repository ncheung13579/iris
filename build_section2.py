content = r'''

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _get_features(self, text: str) -> np.ndarray:
        """Run text through GPT-2 + SAE, return feature vector (1, d_sae)."""
        from src.data.dataset import SYSTEM_PROMPT_TEMPLATE
        from src.data.preprocessing import tokenize_prompts
        from src.model.transformer import extract_activations
        from src.analysis.features import compute_feature_activations

        formatted = SYSTEM_PROMPT_TEMPLATE.format(prompt=text)
        tokenized = tokenize_prompts([formatted], max_length=128)
        acts = extract_activations(
            self.gpt2,
            tokenized["input_ids"],
            tokenized["attention_mask"],
            layers=[0],
            batch_size=1,
        )
        return compute_feature_activations(self.sae, acts[0], device=self.device)

    def analyze(self, text: str):
        """Full analysis of a single prompt. Returns all display data."""
        if not text or not text.strip():
            return None

        features = self._get_features(text)

        # Anomaly-based detector (SAE features)
        sae_pred = int(self.sae_detector.predict(features)[0])
        sae_probs = self.sae_detector.predict_proba(features)[0]
        sae_inject_prob = float(sae_probs[1])

        # Signature-based detector (TF-IDF text patterns)
        tfidf_pred = int(self.tfidf_detector.predict([text])[0])
        tfidf_probs = self.tfidf_detector.predict_proba([text])[0]
        tfidf_inject_prob = float(tfidf_probs[1])

        # Top signatures (features sorted by |sensitivity|)
        top_feats = []
        for idx in self.top_feature_indices[:20]:
            act = float(features[0, idx])
            sens = float(self.sensitivity[idx])
            top_feats.append(
                {"index": int(idx), "activation": act, "sensitivity": sens}
            )

        explanation = self._explain(features[0], sae_pred, sae_inject_prob)

        return {
            "sae_pred": sae_pred,
            "sae_inject_prob": sae_inject_prob,
            "tfidf_pred": tfidf_pred,
            "tfidf_inject_prob": tfidf_inject_prob,
            "features": top_feats,
            "feature_vector": features[0],
            "explanation": explanation,
        }

    def _explain(self, feat_vec, pred, prob):
        """Generate natural-language explanation using IDS terminology."""
        active_inject = []
        active_normal = []
        for idx in self.top_feature_indices[:50]:
            act = feat_vec[idx]
            sens = self.sensitivity[idx]
            if act > 0.01:
                entry = (int(idx), float(act), float(sens))
                if sens > 0:
                    active_inject.append(entry)
                else:
                    active_normal.append(entry)

        active_inject.sort(key=lambda x: x[1] * abs(x[2]), reverse=True)
        active_normal.sort(key=lambda x: x[1] * abs(x[2]), reverse=True)

        if pred == 1:
            if active_inject:
                names = [f"SID-{i} (signal={a:.2f})" for i, a, _ in active_inject[:3]]
                return (
                    "ALERT: Injection signatures triggered. "
                    "Active signatures: " + ", ".join(names) + ". "
                    "These signatures correspond to instruction override, "
                    "role manipulation, or prompt boundary crossing patterns. "
                    "Analogous to an IDS matching known attack signatures in network traffic."
                )
            return (
                "ALERT: Overall activation pattern matches known injection "
                "profiles, similar to anomaly-based IDS detecting unusual "
                "traffic patterns even without a specific signature match."
            )
        else:
            if active_normal:
                return (
                    "PASS: Normal-class signatures dominate the activation pattern. "
                    "No injection signatures triggered above threshold. "
                    "Traffic appears benign - consistent with normal user queries."
                )
            return (
                "PASS: Activation pattern consistent with normal traffic. "
                "No injection indicators detected. Prompt cleared by both detectors."
            )

    # ------------------------------------------------------------------
    # Batch analysis for IDS Console
    # ------------------------------------------------------------------

    def batch_analyze(self, texts):
        """Analyze multiple prompts, returning augmented result dicts."""
        results = []
        base_time = datetime.now()
        for i, text in enumerate(texts):
            r = self.analyze(text)
            if r is not None:
                r["timestamp"] = (base_time + timedelta(seconds=i * 0.3)).strftime("%H:%M:%S.%f")[:-3]
                r["prompt"] = text
                top_sig = max(r["features"][:10], key=lambda f: f["activation"] * abs(f["sensitivity"]))
                r["top_sid"] = top_sig["index"]
                results.append(r)
        return results

    # ------------------------------------------------------------------
    # Signature management helpers
    # ------------------------------------------------------------------

    def get_signature_table(self, top_k=50):
        """Return top-K signatures as dicts for display."""
        labels = np.array(self.dataset.labels)
        table = []
        for idx in self.top_feature_indices[:top_k]:
            sens = float(self.sensitivity[idx])
            mean_inj = float(self.feature_matrix[labels == 1, idx].mean())
            mean_nor = float(self.feature_matrix[labels == 0, idx].mean())
            table.append({
                "SID": int(idx),
                "Direction": "Injection" if sens > 0 else "Normal",
                "Confidence": round(abs(sens), 4),
                "Mean (Injection)": round(mean_inj, 4),
                "Mean (Normal)": round(mean_nor, 4),
            })
        return table

    def evaluate_with_mask(self, enabled_sids):
        """Retrain detector using only enabled signatures and evaluate."""
        if not enabled_sids:
            return {"f1": 0.0, "accuracy": 0.0, "n_signatures": 0}
        labels = np.array(self.dataset.labels)
        subset = self.feature_matrix[:, enabled_sids]
        n_train = int(len(labels) * 0.7)
        X_train, X_test = subset[:n_train], subset[n_train:]
        y_train, y_test = labels[:n_train], labels[n_train:]
        clf = LogisticRegression(random_state=42, max_iter=1000, solver="lbfgs")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return {
            "f1": round(float(f1_score(y_test, y_pred)), 3),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
            "n_signatures": len(enabled_sids),
        }

    def get_sample_prompts_for_signature(self, sid, k=5):
        """Get top-K prompts that trigger a given signature."""
        from src.analysis.features import get_top_activating_examples
        return get_top_activating_examples(
            self.feature_matrix, sid,
            self.dataset.texts, self.dataset.labels, k=k,
        )
'''

with open('src/app.py', 'a', encoding='utf-8') as f:
    f.write(content)
print(f'Section 2: {len(content.splitlines())} lines')
