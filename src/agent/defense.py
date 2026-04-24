"""
Four-layer defense stack for the IRIS agent pipeline.

Implements defense-in-depth with independently toggleable layers:
  Layer 1 (Input):    IRIS SAE-based detection via IRISMiddleware
  Layer 2 (Prompt):   Regex scan for delimiter-breaking / prompt injection patterns
  Layer 3 (Tool):     Risk-level gating based on prior layer results
  Layer 4 (Output):   String matching for system prompt leaks, PII, credentials

Each layer can be enabled/disabled independently, allowing controlled
experiments (e.g., "what gets through with only Layer 1?").

Design: The DefenseStack orchestrates layers but doesn't own the agent.
It wraps the agent's process() method, inserting checks at each stage.

Author: Nathan Cheung
York University | CSSD 2221 | Winter 2026
"""

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.agent.agent import AgentPipeline, AgentResponse
from src.agent.middleware import IRISMiddleware, ScanResult


# ---------------------------------------------------------------------------
# Layer result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LayerResult:
    """Result from a single defense layer.

    Attributes:
        layer_name: Human-readable name (e.g., "Layer 1: IRIS SAE Detection").
        passed: Whether the input passed this layer (True = allowed).
        reason: Why the layer passed or blocked.
        details: Additional data (e.g., matched patterns, risk level).
        latency_ms: Time taken by this layer in milliseconds.
    """
    layer_name: str
    passed: bool
    reason: str
    details: Dict[str, Any]
    latency_ms: float


# ---------------------------------------------------------------------------
# Layer 2: Prompt isolation patterns
# ---------------------------------------------------------------------------

# Patterns that indicate attempts to break out of the user role or inject
# new instructions. These target delimiter injection — a common attack
# vector where the attacker tries to close the user message and open a
# new system/instruction block.
_PROMPT_INJECTION_PATTERNS = [
    # Direct instruction override
    (r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|rules?)", "instruction_override"),
    (r"disregard\s+(?:all\s+)?(?:previous|prior|above)", "instruction_override"),
    (r"forget\s+(?:everything|all|your)\s+(?:previous|prior|above)?", "instruction_override"),
    # Delimiter injection — trying to open a new role block
    (r"\[(?:INST|SYS|SYSTEM)\]", "delimiter_injection"),
    (r"<\|(?:system|im_start|im_end)\|>", "delimiter_injection"),
    (r"###\s*(?:Instruction|System|Human|Assistant)", "delimiter_injection"),
    (r"(?:^|\n)\s*System\s*:", "delimiter_injection"),
    # Role confusion — impersonating system or assistant
    (r"you\s+are\s+now\s+(?:a|an|in)\s+", "role_confusion"),
    (r"(?:new|updated|revised)\s+(?:system\s+)?(?:prompt|instructions?|rules?)\s*:", "role_confusion"),
    (r"(?:switch|change)\s+(?:to|into)\s+(?:\w+\s+)?mode", "role_confusion"),
    # Jailbreak markers
    (r"DAN\s+mode", "jailbreak"),
    (r"do\s+anything\s+now", "jailbreak"),
    (r"(?:act|pretend|behave)\s+as\s+(?:if\s+)?(?:you\s+(?:have|had)\s+)?no\s+(?:restrictions?|rules?|limits?)", "jailbreak"),
]

# Compile patterns once for performance
_COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE | re.MULTILINE), label)
    for pattern, label in _PROMPT_INJECTION_PATTERNS
]


def _check_prompt_isolation(text: str) -> LayerResult:
    """Layer 2: Scan for delimiter-breaking and prompt injection patterns.

    This is a fast, regex-based check that catches common injection
    patterns. It complements the SAE-based detection (Layer 1) by
    catching surface-level patterns the SAE might miss, especially
    for template-based injections.

    Args:
        text: Input text to scan.

    Returns:
        LayerResult with matched patterns in details.
    """
    start = time.time()
    matches = []

    for compiled, label in _COMPILED_PATTERNS:
        match = compiled.search(text)
        if match:
            matches.append({
                "pattern_type": label,
                "matched_text": match.group()[:50],  # Truncate for display
                "position": match.start(),
            })

    elapsed_ms = (time.time() - start) * 1000
    passed = len(matches) == 0

    if passed:
        reason = "No prompt injection patterns detected."
    else:
        types = set(m["pattern_type"] for m in matches)
        reason = f"Detected {len(matches)} injection pattern(s): {', '.join(types)}"

    return LayerResult(
        layer_name="Layer 2: Prompt Isolation",
        passed=passed,
        reason=reason,
        details={"matches": matches, "n_matches": len(matches)},
        latency_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Layer 3: Tool permission check
# ---------------------------------------------------------------------------

# Risk level gating: if prior layers flagged the input, restrict access
# to higher-risk tools.
_RISK_LEVELS = {"low": 0, "medium": 1, "high": 2}


def _check_tool_permission(
    tool_name: Optional[str],
    tool_risk_level: Optional[str],
    prior_results: List[LayerResult],
) -> LayerResult:
    """Layer 3: Gate tool access based on risk level and prior layer results.

    Policy:
      - If all prior layers passed → allow any tool
      - If any prior layer failed → block medium and high risk tools
      - If tool is high risk → always require all prior layers to pass

    Args:
        tool_name: Name of the tool being invoked (None if no tool).
        tool_risk_level: Risk level of the tool ("low"/"medium"/"high").
        prior_results: Results from Layers 1 and 2.

    Returns:
        LayerResult with permission decision.
    """
    start = time.time()

    # No tool being called — pass through
    if tool_name is None:
        elapsed_ms = (time.time() - start) * 1000
        return LayerResult(
            layer_name="Layer 3: Tool Permission",
            passed=True,
            reason="No tool invoked — no permission check needed.",
            details={"tool": None},
            latency_ms=elapsed_ms,
        )

    # Check if any prior layer failed
    any_prior_failed = any(not r.passed for r in prior_results)
    risk_num = _RISK_LEVELS.get(tool_risk_level or "low", 0)

    if any_prior_failed and risk_num >= 1:
        # Prior layers flagged this input — block medium/high risk tools
        elapsed_ms = (time.time() - start) * 1000
        failed_layers = [r.layer_name for r in prior_results if not r.passed]
        return LayerResult(
            layer_name="Layer 3: Tool Permission",
            passed=False,
            reason=(
                f"Tool '{tool_name}' (risk={tool_risk_level}) blocked — "
                f"prior layer(s) flagged input: {', '.join(failed_layers)}"
            ),
            details={
                "tool": tool_name,
                "risk_level": tool_risk_level,
                "blocked_by": failed_layers,
            },
            latency_ms=(time.time() - start) * 1000,
        )

    elapsed_ms = (time.time() - start) * 1000
    return LayerResult(
        layer_name="Layer 3: Tool Permission",
        passed=True,
        reason=f"Tool '{tool_name}' (risk={tool_risk_level}) allowed.",
        details={"tool": tool_name, "risk_level": tool_risk_level},
        latency_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Layer 4: Output scanning
# ---------------------------------------------------------------------------

# Patterns to detect in agent output that could indicate data exfiltration
_OUTPUT_PATTERNS = [
    # System prompt leakage — fragments of common system prompt phrasing
    (r"you are a helpful (?:AI )?assistant", "system_prompt_leak"),
    (r"do not reveal your system prompt", "system_prompt_leak"),
    (r"internal instructions", "system_prompt_leak"),
    # Credential patterns
    (r"(?:api[_-]?key|secret[_-]?key|password|token)\s*[:=]\s*\S+", "credential_leak"),
    (r"sk-[a-zA-Z0-9]{20,}", "credential_leak"),
    (r"ghp_[a-zA-Z0-9]{36}", "credential_leak"),
    (r"AKIA[0-9A-Z]{16}", "credential_leak"),
    # PII patterns (simplified)
    (r"\b\d{3}-\d{2}-\d{4}\b", "ssn_pattern"),
    (r"\b(?:\d{4}[- ]?){3}\d{4}\b", "credit_card_pattern"),
]

_COMPILED_OUTPUT_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), label)
    for pattern, label in _OUTPUT_PATTERNS
]


def _scan_output(text: str, system_prompt: str = "") -> LayerResult:
    """Layer 4: Scan agent output for sensitive data leakage.

    Checks for:
      - System prompt fragments in the output
      - Credential/API key patterns
      - PII patterns (SSN, credit card numbers)

    Args:
        text: Agent's generated output text.
        system_prompt: The actual system prompt (to check for direct leaks).

    Returns:
        LayerResult with detected issues. If issues found, the output
        should be redacted before returning to the user.
    """
    start = time.time()
    findings = []

    # Check for literal system prompt substring leak
    if system_prompt and len(system_prompt) > 20:
        # Check if a significant chunk of the system prompt appears in output
        # Use sliding window of 30 chars to catch partial leaks
        prompt_lower = system_prompt.lower()
        text_lower = text.lower()
        window_size = 30
        for i in range(0, len(prompt_lower) - window_size + 1, 10):
            chunk = prompt_lower[i:i + window_size]
            if chunk in text_lower:
                findings.append({
                    "pattern_type": "system_prompt_leak",
                    "matched_text": chunk[:50],
                    "source": "direct_substring_match",
                })
                break  # One match is enough

    # Check regex patterns
    for compiled, label in _COMPILED_OUTPUT_PATTERNS:
        match = compiled.search(text)
        if match:
            findings.append({
                "pattern_type": label,
                "matched_text": match.group()[:50],
            })

    elapsed_ms = (time.time() - start) * 1000
    passed = len(findings) == 0

    if passed:
        reason = "Output clean — no sensitive data detected."
    else:
        types = set(f["pattern_type"] for f in findings)
        reason = f"REDACT: Detected {len(findings)} sensitive pattern(s): {', '.join(types)}"

    return LayerResult(
        layer_name="Layer 4: Output Scanning",
        passed=passed,
        reason=reason,
        details={"findings": findings, "n_findings": len(findings)},
        latency_ms=elapsed_ms,
    )


def _redact_output(text: str) -> str:
    """Replace sensitive patterns in output with [REDACTED].

    Args:
        text: Agent output that may contain sensitive data.

    Returns:
        Redacted version of the text.
    """
    redacted = text
    for compiled, label in _COMPILED_OUTPUT_PATTERNS:
        redacted = compiled.sub(f"[REDACTED:{label}]", redacted)
    return redacted


# ---------------------------------------------------------------------------
# Defense stack orchestrator
# ---------------------------------------------------------------------------

class DefenseStack:
    """Orchestrates all 4 defense layers around the agent pipeline.

    Each layer can be independently enabled/disabled for controlled
    experiments. The stack processes input through layers 1-3 before
    agent execution, and layer 4 after generation.

    Args:
        agent: The AgentPipeline to protect.
        iris_middleware: IRISMiddleware for Layer 1 (SAE detection).
        enable_layer1: Enable IRIS SAE detection.
        enable_layer2: Enable prompt isolation regex scanning.
        enable_layer3: Enable tool permission gating.
        enable_layer4: Enable output scanning.
    """

    def __init__(
        self,
        agent: AgentPipeline,
        iris_middleware: Optional[IRISMiddleware] = None,
        enable_layer1: bool = True,
        enable_layer2: bool = True,
        enable_layer3: bool = True,
        enable_layer4: bool = True,
    ):
        self.agent = agent
        self.iris_middleware = iris_middleware
        self.layers_enabled = {
            "layer1": enable_layer1,
            "layer2": enable_layer2,
            "layer3": enable_layer3,
            "layer4": enable_layer4,
        }

    def set_layer(self, layer: str, enabled: bool) -> None:
        """Enable or disable a specific layer.

        Args:
            layer: Layer key ("layer1", "layer2", "layer3", "layer4").
            enabled: Whether to enable the layer.
        """
        if layer in self.layers_enabled:
            self.layers_enabled[layer] = enabled

    def process(self, user_input: str) -> AgentResponse:
        """Process a user message through all defense layers and the agent.

        Flow:
          1. Layer 1 (Input):  IRIS SAE scan → BLOCK if threat detected
          2. Layer 2 (Prompt): Regex scan → BLOCK if injection pattern found
          3. Tool dispatch     (agent determines which tool, if any)
          4. Layer 3 (Tool):   Permission check → BLOCK if risk too high
          5. Agent execution   (tool execution + Phi-3 generation)
          6. Layer 4 (Output): Output scan → REDACT if sensitive data found

        If any blocking layer fires, the agent returns a blocked response
        without executing the tool or generating with Phi-3.

        Args:
            user_input: The user's message.

        Returns:
            AgentResponse with defense log populated.
        """
        start = time.time()
        layer_results: List[LayerResult] = []
        blocked = False
        block_reason = None
        threat_score = 0.0

        # ----- Layer 1: IRIS SAE Detection -----
        if self.layers_enabled["layer1"] and self.iris_middleware is not None:
            scan = self.iris_middleware.check(user_input)
            threat_score = scan.probability
            l1_passed = scan.decision != "BLOCK"
            layer_results.append(LayerResult(
                layer_name="Layer 1: IRIS SAE Detection",
                passed=l1_passed,
                reason=scan.explanation,
                details={
                    "decision": scan.decision,
                    "probability": scan.probability,
                    "n_triggered": len(scan.triggered_features),
                    "triggered_features": scan.triggered_features[:5],
                },
                latency_ms=scan.latency_ms,
            ))
            if not l1_passed:
                blocked = True
                block_reason = scan.explanation
        else:
            layer_results.append(LayerResult(
                layer_name="Layer 1: IRIS SAE Detection",
                passed=True,
                reason="Layer disabled.",
                details={"decision": "SKIP"},
                latency_ms=0.0,
            ))

        # ----- Layer 2: Prompt Isolation -----
        if self.layers_enabled["layer2"]:
            l2 = _check_prompt_isolation(user_input)
            layer_results.append(l2)
            if not l2.passed:
                blocked = True
                block_reason = block_reason or l2.reason
        else:
            layer_results.append(LayerResult(
                layer_name="Layer 2: Prompt Isolation",
                passed=True,
                reason="Layer disabled.",
                details={"decision": "SKIP"},
                latency_ms=0.0,
            ))

        # If blocked by input layers, return early without executing
        if blocked:
            elapsed_ms = (time.time() - start) * 1000
            # Add placeholder results for layers 3 and 4
            layer_results.append(LayerResult(
                layer_name="Layer 3: Tool Permission",
                passed=False,
                reason="Skipped — input blocked by prior layer.",
                details={"decision": "SKIP"},
                latency_ms=0.0,
            ))
            layer_results.append(LayerResult(
                layer_name="Layer 4: Output Scanning",
                passed=False,
                reason="Skipped — input blocked by prior layer.",
                details={"decision": "SKIP"},
                latency_ms=0.0,
            ))
            return AgentResponse(
                text=f"[BLOCKED] {block_reason}",
                blocked=True,
                block_reason=block_reason,
                threat_score=threat_score,
                defense_log=[_layer_to_dict(lr) for lr in layer_results],
                latency_ms=elapsed_ms,
            )

        # ----- Tool dispatch (before Layer 3) -----
        dispatch = self.agent.dispatch_tool(user_input)
        tool_name = dispatch[0] if dispatch else None
        tool_risk = None
        if tool_name and tool_name in self.agent.tools:
            tool_risk = self.agent.tools[tool_name].risk_level

        # ----- Layer 3: Tool Permission -----
        if self.layers_enabled["layer3"]:
            l3 = _check_tool_permission(tool_name, tool_risk, layer_results)
            layer_results.append(l3)
            if not l3.passed:
                blocked = True
                block_reason = l3.reason
        else:
            layer_results.append(LayerResult(
                layer_name="Layer 3: Tool Permission",
                passed=True,
                reason="Layer disabled.",
                details={"decision": "SKIP"},
                latency_ms=0.0,
            ))

        if blocked:
            elapsed_ms = (time.time() - start) * 1000
            layer_results.append(LayerResult(
                layer_name="Layer 4: Output Scanning",
                passed=False,
                reason="Skipped — tool access blocked.",
                details={"decision": "SKIP"},
                latency_ms=0.0,
            ))
            return AgentResponse(
                text=f"[BLOCKED] {block_reason}",
                blocked=True,
                block_reason=block_reason,
                threat_score=threat_score,
                defense_log=[_layer_to_dict(lr) for lr in layer_results],
                latency_ms=elapsed_ms,
            )

        # ----- Agent execution -----
        response = self.agent.process(
            user_input,
            defense_results=[_layer_to_dict(lr) for lr in layer_results],
        )

        # ----- Layer 4: Output Scanning -----
        if self.layers_enabled["layer4"]:
            l4 = _scan_output(response.text, self.agent.system_prompt)
            layer_results.append(l4)
            if not l4.passed:
                # Redact sensitive content instead of blocking entirely
                response.text = _redact_output(response.text)
        else:
            layer_results.append(LayerResult(
                layer_name="Layer 4: Output Scanning",
                passed=True,
                reason="Layer disabled.",
                details={"decision": "SKIP"},
                latency_ms=0.0,
            ))

        elapsed_ms = (time.time() - start) * 1000
        response.threat_score = threat_score
        response.defense_log = [_layer_to_dict(lr) for lr in layer_results]
        response.latency_ms = elapsed_ms

        return response


def _layer_to_dict(lr: LayerResult) -> Dict[str, Any]:
    """Convert a LayerResult to a plain dict for JSON serialization."""
    return {
        "layer_name": lr.layer_name,
        "passed": lr.passed,
        "reason": lr.reason,
        "details": lr.details,
        "latency_ms": round(lr.latency_ms, 2),
    }
