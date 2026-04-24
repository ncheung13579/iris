"""
Tool registry for the IRIS agent pipeline.

Implements three sandboxed tools the agent can invoke:
  - read_file: Read files from a restricted sandbox directory
  - calculator: Safe math evaluation (no eval(), uses AST whitelist)
  - lookup_user: Mock user database lookup (the "sensitive" tool)

Design decision: Tools are pure functions wrapped in dataclasses, not
LLM-driven. The agent uses keyword-based dispatch (deterministic and
reliable for a graded demo) rather than asking the LLM to select tools.

Author: Nathan Cheung
York University | CSSD 2221 | Winter 2026
"""

import ast
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from src.agent.sandbox import read_sandboxed_file


# ---------------------------------------------------------------------------
# Safe math evaluator — AST-based, no eval()
# ---------------------------------------------------------------------------

# Whitelist of allowed AST node types for math expressions.
# This prevents code execution while allowing arithmetic.
_ALLOWED_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,  # Python 3.8+ for numeric literals
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
}

# Operator mapping for safe evaluation
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node using only whitelisted operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Non-numeric constant: {node.value!r}")
    elif isinstance(node, ast.BinOp):
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        op_func = _OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        # Guard against excessively large exponents
        if isinstance(node.op, ast.Pow) and right > 1000:
            raise ValueError("Exponent too large (max 1000)")
        return op_func(left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval_node(node.operand)
        op_func = _OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(operand)
    else:
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def safe_calculate(expression: str) -> str:
    """Evaluate a math expression safely using AST parsing.

    Only allows numeric literals and basic arithmetic operators
    (+, -, *, /, //, %, **). No function calls, variable access,
    or code execution.

    Args:
        expression: A math expression string like "2 + 3 * 4".

    Returns:
        String representation of the result.
    """
    expression = expression.strip()
    if not expression:
        return "Error: empty expression"

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        return f"Error: invalid expression — {e}"

    # Verify all nodes are in the whitelist
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            return f"Error: unsupported operation '{type(node).__name__}'"

    try:
        result = _safe_eval_node(tree)
        # Format nicely: drop .0 for integers
        if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except ValueError as e:
        return f"Error: {e}"
    except OverflowError:
        return "Error: result too large"


# ---------------------------------------------------------------------------
# Mock user database — the "sensitive" tool attackers try to abuse
# ---------------------------------------------------------------------------

_MOCK_USERS: Dict[str, Dict[str, str]] = {
    "alice": {
        "name": "Alice Chen",
        "email": "alice.chen@example.com",
        "role": "Engineer",
        "department": "Platform",
    },
    "bob": {
        "name": "Bob Martinez",
        "email": "bob.martinez@example.com",
        "role": "Manager",
        "department": "Product",
    },
    "carol": {
        "name": "Carol Okafor",
        "email": "carol.okafor@example.com",
        "role": "Data Scientist",
        "department": "ML Research",
    },
    "dave": {
        "name": "Dave Kim",
        "email": "dave.kim@example.com",
        "role": "Security Analyst",
        "department": "InfoSec",
    },
    "eve": {
        "name": "Eve Nakamura",
        "email": "eve.nakamura@example.com",
        "role": "Intern",
        "department": "Engineering",
    },
    "admin": {
        "name": "System Administrator",
        "email": "admin@internal.example.com",
        "role": "Administrator",
        "department": "IT Operations",
        "note": "RESTRICTED — admin records require elevated access",
    },
}


def lookup_user(username: str) -> str:
    """Look up a user in the mock database.

    Args:
        username: Username to search for (case-insensitive).

    Returns:
        Formatted user info string, or error message if not found.
    """
    username = username.strip().lower()
    if not username:
        return "Error: please provide a username"

    user = _MOCK_USERS.get(username)
    if user is None:
        available = ", ".join(sorted(_MOCK_USERS.keys()))
        return f"User '{username}' not found. Available users: {available}"

    lines = [f"User record for '{username}':"]
    for key, value in user.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool dataclass and registry
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """A tool the agent can invoke.

    Attributes:
        name: Unique identifier used in dispatch.
        description: Human-readable description (shown in system prompt).
        execute: Function that takes a string argument and returns a string.
        risk_level: Security classification — low/medium/high.
            Affects whether the defense stack allows execution.
    """
    name: str
    description: str
    execute: Callable[[str], str]
    risk_level: str  # "low", "medium", "high"


def build_tool_registry(sandbox_root: Optional[Path] = None) -> Dict[str, Tool]:
    """Create the default tool registry with sandboxed file access.

    Args:
        sandbox_root: Root directory for file access. Defaults to
            data/agent_sandbox/ relative to the project root.

    Returns:
        Dict mapping tool name to Tool instance.
    """
    if sandbox_root is None:
        # Default: project_root/data/agent_sandbox/
        sandbox_root = Path(__file__).resolve().parent.parent.parent / "data" / "agent_sandbox"

    def _read_file(path: str) -> str:
        try:
            return read_sandboxed_file(path, sandbox_root)
        except PermissionError as e:
            return f"ACCESS DENIED: {e}"
        except FileNotFoundError as e:
            return f"FILE NOT FOUND: {e}"

    return {
        "read_file": Tool(
            name="read_file",
            description="Read a file from the workspace. Only files in the sandbox directory are accessible.",
            execute=_read_file,
            risk_level="low",
        ),
        "calculator": Tool(
            name="calculator",
            description="Evaluate a math expression. Supports +, -, *, /, //, %, **.",
            execute=safe_calculate,
            risk_level="low",
        ),
        "lookup_user": Tool(
            name="lookup_user",
            description="Look up a user's information by username.",
            execute=lookup_user,
            risk_level="medium",
        ),
    }
