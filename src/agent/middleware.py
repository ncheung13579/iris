"""
IRIS middleware — wraps the existing IRISPipeline for real-time scan decisions.

Translates IRISPipeline's analysis output into actionable security decisions
(BLOCK / WARN / PASS) with configurable thresholds. Maintains an audit trail
for the IDS console.

The middleware operates on a separate model (GPT-2 + SAE) from the application
model (Phi-3), analogous to a network IDS that inspects traffic independently
of the application it protects.

Author: Nathan Cheung (ncheung3@my.yorku.ca)
York University | CSSD 2221 | Winter 2026
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScanResult:
    """Result of an IRIS middleware scan on a single input.

    Attributes:
        decision: One of "BLOCK", "WARN", "PASS".
        probability: Injection probability from the SAE detector (0.0–1.0).
        triggered_features: List of top SAE features that fired.
        explanation: Human-readable explanation of the decision.
        latency_ms: Time taken for the scan in milliseconds.
    """
    decision: str  # "BLOCK", "WARN", "PASS"
    probability: float
    triggered_features: List[Dict[str, float]]
    explanation: str
    latency_ms: float


class IRISMiddleware:
    """Wraps IRISPipeline to provide real-time BLOCK/WARN/PASS decisions.

    Two thresholds divide the probability space:
      - probability >= block_threshold  → BLOCK
      - probability >= warn_threshold   → WARN
      - otherwise                       → PASS

    Args:
        pipeline: A loaded IRISPipeline instance (from src.app).
        block_threshold: Probability above which inputs are blocked.
        warn_threshold: Probability above which inputs trigger warnings.
    """

    def __init__(
        self,
        pipeline: Any,
        block_threshold: float = 0.85,
        warn_threshold: float = 0.4,
    ):
        self.pipeline = pipeline
        self.block_threshold = block_threshold
        self.warn_threshold = warn_threshold
        self._log: List[Dict[str, Any]] = []
        # Use the agent-mode detector (top-K feature selection) if available,
        # otherwise fall back to the default SAE detector probability.
        self._agent_detector = getattr(pipeline, "agent_detector", None)
        self._detect_features = getattr(pipeline, "_detect_features", None)

    def check(self, text: str) -> ScanResult:
        """Run IRIS analysis on the input text and return a scan decision.

        Args:
            text: User input to scan.

        Returns:
            ScanResult with decision, probability, and triggered features.
        """
        start = time.time()

        result = self.pipeline.analyze(text)
        elapsed_ms = (time.time() - start) * 1000

        if result is None:
            return ScanResult(
                decision="PASS",
                probability=0.0,
                triggered_features=[],
                explanation="Empty input — no threat detected.",
                latency_ms=elapsed_ms,
            )

        # Use agent detector (top-K feature selection) for calibrated probabilities
        if self._agent_detector is not None:
            import numpy as np
            fv = result["feature_vector"].reshape(1, -1)
            if self._detect_features is not None:
                fv = self._detect_features(fv)
            prob = float(self._agent_detector.predict_proba(fv)[0, 1])
        else:
            prob = result["sae_inject_prob"]
        features = result.get("features", [])

        # Only include features that are actually firing (activation > 0.01)
        triggered = [
            {"index": f["index"], "activation": f["activation"], "sensitivity": f["sensitivity"]}
            for f in features
            if f["activation"] > 0.01 and f["sensitivity"] > 0
        ]

        if prob >= self.block_threshold:
            decision = "BLOCK"
            explanation = (
                f"ALERT: Injection probability {prob:.1%} exceeds block threshold "
                f"({self.block_threshold:.1%}). {len(triggered)} injection signatures triggered."
            )
        elif prob >= self.warn_threshold:
            decision = "WARN"
            explanation = (
                f"WARNING: Injection probability {prob:.1%} exceeds warn threshold "
                f"({self.warn_threshold:.1%}). Proceeding with elevated monitoring."
            )
        else:
            decision = "PASS"
            explanation = (
                f"CLEAR: Injection probability {prob:.1%} below thresholds. "
                f"No injection signatures triggered."
            )

        scan_result = ScanResult(
            decision=decision,
            probability=prob,
            triggered_features=triggered,
            explanation=explanation,
            latency_ms=elapsed_ms,
        )

        # Audit trail entry
        self._log.append({
            "text_preview": text[:80] + ("..." if len(text) > 80 else ""),
            "decision": decision,
            "probability": prob,
            "n_triggered": len(triggered),
            "latency_ms": elapsed_ms,
        })

        return scan_result

    @property
    def log(self) -> List[Dict[str, Any]]:
        """Return the audit trail of all scans performed."""
        return list(self._log)

    def clear_log(self) -> None:
        """Clear the audit trail."""
        self._log.clear()
