"""
IRIS Agent Pipeline — Defended AI agent with interpretability-guided security.

This module implements a functional AI agent (Phi-3-mini) protected by IRIS
as middleware defense. The architecture separates the security sensor (GPT-2
+ SAE) from the application model (Phi-3), analogous to a network IDS that
monitors traffic independently of the application it protects.

Components:
    AgentPipeline: Core agent with tool dispatch and Phi-3 generation
    IRISMiddleware: Wraps IRISPipeline for real-time scan decisions
    DefenseStack: Orchestrates 4-layer defense-in-depth
"""

from src.agent.agent import AgentPipeline, AgentResponse
from src.agent.defense import DefenseStack, LayerResult
from src.agent.middleware import IRISMiddleware, ScanResult

__all__ = [
    "AgentPipeline",
    "AgentResponse",
    "IRISMiddleware",
    "ScanResult",
    "DefenseStack",
    "LayerResult",
]
