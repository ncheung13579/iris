"""
Sandboxed file access for the IRIS agent pipeline.

Prevents directory traversal attacks by resolving paths and ensuring
they remain within the allowed sandbox root. This is the agent's
equivalent of chroot — the tool can only see files we explicitly allow.

Author: Nathan Cheung
York University | CSSD 2221 | Winter 2026
"""

from pathlib import Path
from typing import Optional


def validate_path(path: str, allowed_root: Path) -> Path:
    """Resolve a user-supplied path and verify it stays within the sandbox.

    Uses Path.resolve() to canonicalize the path (eliminating .., symlinks,
    etc.) then checks that the resolved path starts with the allowed root.
    This prevents directory traversal attacks like "../../etc/passwd".

    Args:
        path: User-supplied file path (may be relative or absolute).
        allowed_root: The sandbox root directory. All access must stay
            within this directory tree.

    Returns:
        The resolved, validated Path object.

    Raises:
        PermissionError: If the resolved path escapes the sandbox.
        FileNotFoundError: If the resolved path does not exist.
    """
    allowed_root = allowed_root.resolve()
    # Join relative paths against the sandbox root
    if not Path(path).is_absolute():
        resolved = (allowed_root / path).resolve()
    else:
        resolved = Path(path).resolve()

    # Security check: resolved path must be inside the sandbox
    if not str(resolved).startswith(str(allowed_root)):
        raise PermissionError(
            f"Access denied: path '{path}' resolves outside the sandbox. "
            f"Allowed root: {allowed_root}"
        )

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")

    if not resolved.is_file():
        raise PermissionError(f"Not a file: {resolved}")

    return resolved


def read_sandboxed_file(path: str, allowed_root: Path, max_chars: int = 2000) -> str:
    """Read a file from the sandbox, with path validation and size limits.

    Args:
        path: User-supplied file path.
        allowed_root: Sandbox root directory.
        max_chars: Maximum characters to return (truncates with notice).

    Returns:
        File contents as a string, truncated if necessary.

    Raises:
        PermissionError: If path escapes sandbox or is not a file.
        FileNotFoundError: If file does not exist.
    """
    validated = validate_path(path, allowed_root)
    content = validated.read_text(encoding="utf-8", errors="replace")

    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[Truncated — showing first {max_chars} characters]"

    return content
