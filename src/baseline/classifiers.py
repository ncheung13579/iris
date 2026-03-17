"""
Classical ML baselines for prompt injection detection.

This module implements the baseline classifiers described in Design Document
Section 4.4 (Detection Pipeline). These baselines serve as the comparison
points for the SAE-based detector — if the SAE features don't outperform
these simpler approaches, the decomposition hasn't found useful structure.

Three tiers of baseline:
  1. Text-level: TF-IDF features → classical classifiers (no neural model)
  2. Raw activations: Logistic regression on 768-dim residual stream vectors
  3. SAE features: Logistic regression on 6144-dim (or top-K) SAE features

The comparison between tiers 2 and 3 is the core experiment (C3): it tests
whether the SAE decomposition adds value beyond what's already present in
the raw activation space.

Author: Nathan Cheung (ncheung3@my.yorku.ca)
York University | CSSD 2221 | Winter 2026
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def train_tfidf_baseline(
    texts: List[str],
    labels: List[int],
    seed: int = 42,
) -> Tuple[Pipeline, Pipeline]:
    """
    Train TF-IDF-based text classifiers for prompt injection detection.

    Builds two pipelines — Logistic Regression and Random Forest — both
    using the same TF-IDF feature representation. These are the "classical
    baseline" from Design Document Section 4.4 (detection approach #1).

    Args:
        texts: List of raw prompt strings (training set only).
        labels: Corresponding integer labels (0 = normal, 1 = injection).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (logistic_regression_pipeline, random_forest_pipeline),
        both fitted on the provided data.
    """
    # --- TF-IDF configuration rationale ---
    # max_features=5000: caps vocabulary size to prevent overfitting on rare
    #   tokens. 5000 is a standard starting point for short-text classification
    #   — large enough to capture important terms, small enough to fit in memory.
    # ngram_range=(1, 2): includes bigrams alongside unigrams. This is critical
    #   for injection detection because many injection patterns are multi-word
    #   phrases: "ignore previous", "disregard above", "new instructions",
    #   "you are now". Unigrams alone would miss these compositional signals.
    # sublinear_tf=True: applies log(1 + tf) scaling. Without this, a word
    #   repeated 10 times gets 10x the weight of one that appears once, which
    #   disproportionately rewards repetitive text over informative terms.
    tfidf_params = {
        "max_features": 5000,
        "ngram_range": (1, 2),
        "sublinear_tf": True,
    }

    # --- Logistic Regression pipeline ---
    # Why logistic regression? It produces calibrated probabilities and a
    # weight vector that shows which TF-IDF features matter most. This makes
    # it directly comparable to the SAE-based logistic regression detector.
    # max_iter=1000: increased from the default 100 because TF-IDF feature
    # spaces can require more iterations to converge, especially with bigrams.
    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", LogisticRegression(
            random_state=seed,
            max_iter=1000,
            solver="lbfgs",
        )),
    ])
    lr_pipeline.fit(texts, labels)

    # --- Random Forest pipeline ---
    # Why random forest as a second baseline? It captures non-linear
    # interactions between features that logistic regression misses. If RF
    # substantially outperforms LR on text features, that suggests non-linear
    # patterns in the injection data that a linear SAE detector might also miss.
    # n_estimators=200: enough trees for stable predictions without excessive
    # training time on a 5000-feature space.
    rf_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,  # Use all CPU cores — harmless on Colab
        )),
    ])
    rf_pipeline.fit(texts, labels)

    return lr_pipeline, rf_pipeline


def evaluate_classifier(
    pipeline: object,
    texts_or_features: object,
    labels: List[int],
    name: str = "",
) -> Dict[str, float]:
    """
    Evaluate a fitted classifier and print a formatted report.

    Works with both sklearn Pipelines (text input) and plain estimators
    (array input), since the raw-activation and SAE-feature baselines
    receive numpy arrays rather than text.

    Args:
        pipeline: A fitted sklearn Pipeline or estimator with .predict()
                  and .predict_proba() methods.
        texts_or_features: Input data — either a list of strings (for
                          TF-IDF pipelines) or a numpy array of features
                          (for activation/SAE baselines).
        labels: Ground truth integer labels.
        name: Display name for the classifier (used in printed output).

    Returns:
        Dict with keys: accuracy, precision, recall, f1, roc_auc.
    """
    y_true = np.array(labels)
    y_pred = pipeline.predict(texts_or_features)

    # ROC-AUC requires probability estimates, not hard predictions.
    # We use the probability of the positive class (injection = 1).
    # If predict_proba isn't available (rare for our classifiers), we
    # fall back to the hard prediction accuracy for AUC.
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(texts_or_features)[:, 1]
        roc_auc = roc_auc_score(y_true, y_prob)
    else:
        roc_auc = roc_auc_score(y_true, y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
    }

    # Print a formatted summary so notebooks get readable output
    header = f"=== {name} ===" if name else "=== Classifier Evaluation ==="
    print(header)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print()
    # Full sklearn report for detailed per-class breakdown
    print(classification_report(
        y_true, y_pred,
        target_names=["normal", "injection"],
        zero_division=0,
    ))

    return metrics


def train_activation_baseline(
    activations: np.ndarray,
    labels: List[int],
    seed: int = 42,
) -> LogisticRegression:
    """
    Train a logistic regression on raw transformer activations.

    This is detection approach #2 from Design Document Section 4.4.
    It tests whether a simple linear classifier on the 768-dim residual
    stream can already detect injections — no SAE needed. If this baseline
    performs well, the SAE decomposition must *exceed* it to justify its
    complexity. If it performs poorly, the activation space doesn't linearly
    separate injection from normal prompts, and the SAE's non-linear
    encoding (via ReLU) may find structure that raw activations obscure.

    Args:
        activations: Numpy array of shape (n_samples, 768) — residual
                     stream activations at the final token position.
        labels: Corresponding integer labels (0 = normal, 1 = injection).
        seed: Random seed for reproducibility.

    Returns:
        Fitted LogisticRegression model.
    """
    # Why logistic regression specifically (not SVM, not a neural net)?
    # 1. Matches the SAE detector's classifier, so the only difference
    #    between this baseline and the SAE detector is the feature space.
    #    This makes the comparison fair — any performance gap comes from
    #    the features, not the classifier.
    # 2. The learned weights are interpretable: each of the 768 dimensions
    #    gets a coefficient, showing which activation dimensions matter.
    # max_iter=1000: 768-dim input can take more iterations than default.
    # C=1.0 (default): standard L2 regularization strength.
    model = LogisticRegression(
        random_state=seed,
        max_iter=1000,
        solver="lbfgs",
    )
    model.fit(activations, labels)
    return model


def train_sae_feature_baseline(
    feature_matrix: np.ndarray,
    labels: List[int],
    seed: int = 42,
    top_k: Optional[int] = None,
) -> LogisticRegression:
    """
    Train a logistic regression on SAE feature activation vectors.

    This is detection approach #3 from Design Document Section 4.4 — the
    core experiment. If this classifier outperforms the raw activation
    baseline (approach #2), the SAE decomposition has found structure that
    helps detection. If it doesn't, the SAE features are no more useful
    than the raw residual stream for this task.

    The top_k parameter enables the ablation study from Experiment A2:
    how many SAE features do you need before the detector matches or
    exceeds raw-activation performance? If a small subset (e.g., top 50
    out of 6144) suffices, that's strong evidence the SAE has isolated
    specific injection-relevant features rather than diffusely spreading
    information across all features.

    Args:
        feature_matrix: Numpy array of shape (n_samples, d_sae) where
                        d_sae is typically 6144 (8x expansion of 768).
                        Each row is the SAE encoder output for one prompt.
        labels: Corresponding integer labels (0 = normal, 1 = injection).
        seed: Random seed for reproducibility.
        top_k: If provided, only use the first top_k columns of the
               feature matrix. The caller is responsible for sorting
               columns by injection sensitivity before passing them in.

    Returns:
        Fitted LogisticRegression model.
    """
    # Subset to top-K features if requested.
    # Why does the caller sort columns rather than this function?
    # Because injection-sensitivity scoring is computed in the analysis
    # module (src/analysis/features.py), and this baseline module should
    # not depend on analysis — that would violate the dependency graph
    # in CLAUDE.md (baseline depends only on data, not on sae/analysis).
    if top_k is not None:
        feature_matrix = feature_matrix[:, :top_k]

    # Same classifier as the raw activation baseline — logistic regression
    # with identical hyperparameters. This ensures the comparison is fair:
    # any difference in detection performance comes entirely from the
    # feature representation (SAE features vs. raw activations), not from
    # the classifier or its tuning.
    model = LogisticRegression(
        random_state=seed,
        max_iter=1000,
        solver="lbfgs",
    )
    model.fit(feature_matrix, labels)
    return model
