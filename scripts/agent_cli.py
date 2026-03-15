"""
Interactive CLI for the IRIS defended agent pipeline.

Provides a terminal-based interface with colored output:
  - Green: PASS (allowed through all layers)
  - Red:   BLOCK (stopped by a defense layer)
  - Yellow: WARNING (elevated monitoring)

Usage:
    python -m scripts.agent_cli [--no-phi3] [--threshold 0.5] [--quantize]

The --no-phi3 flag runs in "detection-only" mode (no response generation),
useful for testing defense layers without loading the 3.8B model.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


# ANSI color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _colored(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


def _print_layer_result(layer: dict) -> None:
    """Print a single defense layer result with color coding."""
    if layer.get("details", {}).get("decision") == "SKIP":
        icon = "⊘"
        color = Colors.GRAY
    elif layer["passed"]:
        icon = "✓"
        color = Colors.GREEN
    else:
        icon = "✗"
        color = Colors.RED

    name = layer["layer_name"]
    reason = layer["reason"]
    latency = layer.get("latency_ms", 0)
    print(f"  {_colored(icon, color)} {_colored(name, Colors.BOLD)}")
    print(f"    {_colored(reason, color)} [{latency:.1f}ms]")


def main():
    parser = argparse.ArgumentParser(description="IRIS Agent CLI")
    parser.add_argument(
        "--no-phi3", action="store_true",
        help="Detection-only mode (skip Phi-3 loading)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="IRIS detection threshold (default: 0.5)",
    )
    parser.add_argument(
        "--quantize", action="store_true",
        help="Use 4-bit quantization for Phi-3 (saves VRAM)",
    )
    args = parser.parse_args()

    print(_colored("=" * 60, Colors.CYAN))
    print(_colored("  IRIS Defended Agent Pipeline — Interactive CLI", Colors.BOLD))
    print(_colored("=" * 60, Colors.CYAN))
    print()

    # Load IRIS detection pipeline (GPT-2 + SAE)
    print("Loading IRIS detection engine...")
    from src.app import IRISPipeline
    iris_pipeline = IRISPipeline(project_root=str(project_root))
    iris_pipeline.load()

    from src.agent.middleware import IRISMiddleware
    from src.agent.tools import build_tool_registry
    from src.agent.defense import DefenseStack

    middleware = IRISMiddleware(
        iris_pipeline,
        block_threshold=args.threshold,
    )
    tools = build_tool_registry()

    if args.no_phi3:
        # Detection-only mode: create a minimal agent that echoes tool results
        print(_colored("Running in detection-only mode (no Phi-3).", Colors.YELLOW))
        from src.agent.agent import AgentPipeline, AgentResponse

        class EchoAgent(AgentPipeline):
            """Minimal agent that returns tool results without LLM generation."""
            def __init__(self, tools):
                self.tools = tools
                self.system_prompt = "Detection-only mode."

            def dispatch_tool(self, user_input):
                return super().dispatch_tool(user_input)

            def process(self, user_input, defense_results=None):
                dispatch = self.dispatch_tool(user_input)
                if dispatch:
                    tool_name, tool_arg = dispatch
                    tool_output = self.execute_tool(tool_name, tool_arg)
                    text = f"[Tool: {tool_name}] {tool_output}"
                else:
                    text = "[No Phi-3 loaded — detection-only mode]"
                return AgentResponse(
                    text=text,
                    tool_called=dispatch[0] if dispatch else None,
                    tool_input=dispatch[1] if dispatch else None,
                    defense_log=defense_results or [],
                )

        agent = EchoAgent(tools)
    else:
        # Full mode: load Phi-3 for response generation
        from src.agent.agent import AgentPipeline, load_phi3
        phi3_model, phi3_tokenizer = load_phi3(quantize_4bit=args.quantize)
        agent = AgentPipeline(phi3_model, phi3_tokenizer, tools)

    defense = DefenseStack(agent, iris_middleware=middleware)

    print()
    print(_colored("Ready! Type a message or 'quit' to exit.", Colors.GREEN))
    print(_colored("Commands: /layers, /toggle <N>, /threshold <val>", Colors.GRAY))
    print()

    while True:
        try:
            user_input = input(_colored("You > ", Colors.BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "/quit"):
            print("Goodbye!")
            break

        # CLI commands
        if user_input.startswith("/layers"):
            for key, enabled in defense.layers_enabled.items():
                status = _colored("ON", Colors.GREEN) if enabled else _colored("OFF", Colors.RED)
                print(f"  {key}: {status}")
            continue

        if user_input.startswith("/toggle"):
            parts = user_input.split()
            if len(parts) == 2:
                layer_key = f"layer{parts[1]}"
                if layer_key in defense.layers_enabled:
                    new_val = not defense.layers_enabled[layer_key]
                    defense.set_layer(layer_key, new_val)
                    status = _colored("ON", Colors.GREEN) if new_val else _colored("OFF", Colors.RED)
                    print(f"  {layer_key}: {status}")
                else:
                    print(f"  Unknown layer: {parts[1]} (use 1-4)")
            continue

        if user_input.startswith("/threshold"):
            parts = user_input.split()
            if len(parts) == 2:
                try:
                    new_thresh = float(parts[1])
                    middleware.block_threshold = new_thresh
                    print(f"  Threshold set to {new_thresh}")
                except ValueError:
                    print("  Invalid threshold value")
            continue

        # Process through defense stack
        response = defense.process(user_input)

        # Print defense log
        print()
        print(_colored("─── Defense Log ───", Colors.CYAN))
        for layer in response.defense_log:
            _print_layer_result(layer)

        # Print response
        print()
        if response.blocked:
            print(_colored("─── BLOCKED ───", Colors.RED))
            print(_colored(response.text, Colors.RED))
        else:
            if response.tool_called:
                print(_colored(f"─── Tool: {response.tool_called}({response.tool_input}) ───", Colors.CYAN))
                print(_colored(response.tool_output or "", Colors.GRAY))
            print(_colored("─── Response ───", Colors.GREEN))
            print(response.text)

        print(_colored(f"\n[Threat: {response.threat_score:.1%} | Latency: {response.latency_ms:.0f}ms]", Colors.GRAY))
        print()


if __name__ == "__main__":
    main()
