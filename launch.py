#!/usr/bin/env python3
"""
IRIS --Neural IDS for LLM Agent Pipelines
==========================================

One-command launcher. Run this script to install dependencies,
verify checkpoints, and launch the interactive dashboard.

Usage:
    python launch.py

Author: Nathan Cheung ()
York University | CSSD 2221 | Winter 2026
"""

import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

# Suppress noisy third-party deprecation warnings globally
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*Blocks constructor.*")

MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 12)

# ── Terminal colors (graceful fallback if not supported) ──────────
try:
    os.system("")  # Enable ANSI on Windows
except Exception:
    pass

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


BANNER = f"""
{CYAN}{BOLD}  ___  ____   ___  ____
 |_ _||  _ \\ |_ _|/ ___|
  | | | |_) | | | \\___ \\
  | | |  _ <  | |  ___) |
 |___||_| \\_\\|___||____/{RESET}

  {BOLD}Neural IDS for LLM Agent Pipelines{RESET}
  {DIM}Interpretability Research for Injection Security{RESET}
  {DIM}Nathan Cheung (){RESET}
  {DIM}York University | CSSD 2221 | Winter 2026{RESET}
"""

SPINNER_CHARS = "|/-\\"


def elapsed_str(seconds: float) -> str:
    """Format elapsed time nicely."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def size_str(nbytes: int) -> str:
    """Format file size."""
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    elif nbytes < 1024 * 1024 * 1024:
        return f"{nbytes / (1024 * 1024):.1f} MB"
    else:
        return f"{nbytes / (1024 * 1024 * 1024):.2f} GB"


def step_header(num: int, total: int, title: str, duration: float = None) -> None:
    """Print a styled step header, or a step completion footer."""
    if duration is not None:
        print(
            f"  {GREEN}{BOLD}[{num}/{total}]{RESET} {BOLD}{title}{RESET} "
            f"{DIM}-- done in {elapsed_str(duration)}{RESET}"
        )
    else:
        print(f"\n  {CYAN}{BOLD}[{num}/{total}]{RESET} {BOLD}{title}{RESET}")
        print(f"  {DIM}{'-' * 52}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}OK{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}--{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}FAIL{RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {DIM}  {msg}{RESET}")


def check_python() -> None:
    """Verify Python version is compatible."""
    v = sys.version_info[:2]
    if v < MIN_PYTHON or v > MAX_PYTHON:
        fail(
            f"Python {v[0]}.{v[1]} detected --"
            f"IRIS requires {MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
            f"-{MAX_PYTHON[0]}.{MAX_PYTHON[1]}"
        )
        print()
        info(f"Interpreter: {sys.executable}")
        info(
            f"Install Python 3.10-3.12 from https://www.python.org/downloads/"
        )
        info(f"Then re-run:  python3.12 launch.py")
        sys.exit(1)
    ok(f"Python {v[0]}.{v[1]}.{sys.version_info[2]} {DIM}({sys.executable}){RESET}")


def install_dependencies(root: Path) -> None:
    """Install pip dependencies with a live spinner and stopwatch."""
    import threading

    reqs = root / "requirements.txt"
    n_packages = sum(
        1 for line in reqs.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    info(f"Installing {n_packages} packages from requirements.txt")
    info(f"This may take several minutes on first run (PyTorch is ~2 GB)")

    t0 = time.time()
    spinner = "|/-\\"
    result_holder = [None]
    phase = ["Resolving dependencies"]

    def run_pip():
        """Run pip in a background thread, updating phase as it goes."""
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "pip", "install",
                "-r", str(reqs),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            stripped = line.strip()
            if not stripped:
                continue
            if "Collecting" in stripped:
                pkg = stripped.replace("Collecting ", "").split()[0]
                phase[0] = f"Collecting {pkg}"
            elif "Downloading" in stripped:
                # Extract filename and size, e.g. "torch-2.1.0-..whl (200 MB)"
                parts = stripped.replace("Downloading ", "").strip()
                # Get just the filename (last path segment) and size
                url_and_size = parts.split()
                fname = url_and_size[0].split("/")[-1] if url_and_size else parts
                size = url_and_size[1] if len(url_and_size) > 1 else ""
                # Trim long filenames
                if len(fname) > 40:
                    fname = fname[:37] + "..."
                phase[0] = f"Downloading {fname} {size}"
            elif "Installing collected" in stripped:
                phase[0] = "Installing packages"
            elif "already satisfied" in stripped.lower():
                pkg = stripped.split("Requirement already satisfied: ")[-1].split()[0]
                phase[0] = f"Checking {pkg}"
        proc.wait()
        result_holder[0] = proc.returncode

    pip_thread = threading.Thread(target=run_pip, daemon=True)
    pip_thread.start()

    # Show spinner + stopwatch + current phase while pip runs
    i = 0
    while pip_thread.is_alive():
        elapsed = time.time() - t0
        char = spinner[i % len(spinner)]
        status = phase[0]
        if len(status) > 50:
            status = status[:47] + "..."
        print(
            f"\r  {CYAN}{char}{RESET} {status:<52} "
            f"{DIM}[{elapsed_str(elapsed)}]{RESET}",
            end="", flush=True,
        )
        i += 1
        time.sleep(0.15)

    pip_thread.join()
    dt = time.time() - t0

    # Clear the spinner line
    print(f"\r{' ' * 78}\r", end="")

    if result_holder[0] != 0:
        fail(f"Dependency installation failed (exit code {result_holder[0]})")
        info("Try manually:  pip install -r requirements.txt")
        sys.exit(1)

    ok(f"Dependencies ready in {elapsed_str(dt)}")


def verify_checkpoints(root: Path) -> None:
    """Check that all required model files exist."""
    required = {
        "checkpoints/sae_d10240_lambda1e-04.pt": (
            "Sparse Autoencoder (10240-dim, trained on GPT-2 Large layer 29)",
            "notebook 04",
        ),
        "checkpoints/sensitivity_scores.npy": (
            "Injection-sensitivity scores (10240 signature weights)",
            "notebook 05",
        ),
        "checkpoints/feature_matrix.npy": (
            "Feature activation matrix (1000 prompts x 10240 features)",
            "notebook 05",
        ),
        "data/processed/iris_dataset_balanced.json": (
            "Curated dataset (500 normal + 500 injection prompts)",
            "notebook 01",
        ),
        "results/metrics/j2_evaluation.json": (
            "SAE evaluation metrics (target layer, train config)",
            "notebook 02",
        ),
        "results/metrics/c3_detection_comparison.json": (
            "Detection pipeline comparison (F1, AUC for all approaches)",
            "notebook 06",
        ),
        "results/metrics/c4_adversarial_evasion.json": (
            "Adversarial evasion rates per strategy",
            "notebook 07",
        ),
        "results/metrics/defense_v2.json": (
            "Defense v1 vs v2 comparison (evasion reduction)",
            "notebook 16",
        ),
    }

    total_size = 0
    missing = []

    for fpath, (desc, source) in required.items():
        p = root / fpath
        if p.exists():
            sz = p.stat().st_size
            total_size += sz
            ok(f"{fpath} {DIM}({size_str(sz)}){RESET}")
            info(desc)
        else:
            fail(f"{fpath} --{RED}MISSING{RESET}")
            info(f"{desc}")
            info(f"Generate with: {source}")
            missing.append(fpath)

    if missing:
        print()
        fail(f"{len(missing)} required file(s) missing")
        info("Run the research notebooks on Google Colab first,")
        info("then copy the output files into this directory.")
        sys.exit(1)

    print()
    ok(f"All artifacts verified --{size_str(total_size)} total")


def load_engine(root: Path) -> object:
    """Import and initialize the IRIS pipeline with progress."""
    sys.path.insert(0, str(root))

    # -- Import phase (suppress noisy library output during import) --
    import io

    class _QuietStream:
        """Captures library print() calls so our output stays clean."""
        def __init__(self):
            self.captured = io.StringIO()
        def write(self, s):
            self.captured.write(s)
        def flush(self):
            pass

    _real_stdout = sys.stdout
    _real_stderr = sys.stderr

    t0 = time.time()
    _real_stdout.write(f"  {DIM}  Importing libraries...{RESET}")
    _real_stdout.flush()
    sys.stdout = _QuietStream()
    sys.stderr = _QuietStream()
    from src.app import IRISPipeline, build_app  # noqa: F811
    sys.stdout = _real_stdout
    sys.stderr = _real_stderr
    dt = time.time() - t0
    print(f"\r  {GREEN}OK{RESET} Libraries imported {DIM}({elapsed_str(dt)}){RESET}")

    pipeline = IRISPipeline(str(root))

    # -- Load each component with timing --
    from src.sae.architecture import SparseAutoencoder  # noqa: F401
    from src.data.dataset import IrisDataset
    from src.model.transformer import load_model
    from src.baseline.classifiers import (
        train_tfidf_baseline,
        train_sae_feature_baseline,
    )
    import numpy as np
    import torch
    import json

    # 1. Dataset
    t0 = time.time()
    _real_stdout.write(f"  {DIM}  Loading dataset...{RESET}")
    _real_stdout.flush()
    sys.stdout = _QuietStream()
    pipeline.dataset = IrisDataset.load(
        pipeline.root / "data/processed/iris_dataset_balanced.json"
    )
    sys.stdout = _real_stdout
    dt = time.time() - t0
    n = len(pipeline.dataset.texts)
    print(
        f"\r  {GREEN}OK{RESET} Dataset loaded: "
        f"{BOLD}{n}{RESET} prompts {DIM}({elapsed_str(dt)}){RESET}"
    )

    # 2. SAE checkpoint
    t0 = time.time()
    ckpt_path = pipeline.root / "checkpoints/sae_d10240_lambda1e-04.pt"
    ckpt_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(
        f"  {DIM}  Loading SAE checkpoint ({ckpt_mb:.0f} MB)...{RESET}",
        end="", flush=True,
    )
    ckpt = torch.load(ckpt_path, map_location=pipeline.device)
    cfg = ckpt["config"]
    from src.sae.architecture import SparseAutoencoder
    pipeline.sae = SparseAutoencoder(
        d_input=cfg["d_input"],
        expansion_factor=cfg["expansion_factor"],
        sparsity_coeff=cfg.get("sparsity_coeff", 1e-4),
    )
    pipeline.sae.load_state_dict(ckpt["model_state_dict"])
    pipeline.sae = pipeline.sae.to(pipeline.device).eval()
    dt = time.time() - t0
    d_sae = cfg["d_input"] * cfg["expansion_factor"]
    print(
        f"\r  {GREEN}OK{RESET} SAE loaded: "
        f"{BOLD}{d_sae}{RESET} features "
        f"({cfg['d_input']}x{cfg['expansion_factor']}) "
        f"{DIM}({elapsed_str(dt)}){RESET}"
    )

    # 3. Sensitivity scores + feature matrix
    t0 = time.time()
    print(f"  {DIM}  Loading detection signatures...{RESET}", end="", flush=True)
    pipeline.sensitivity = np.load(
        pipeline.root / "checkpoints/sensitivity_scores.npy"
    )
    pipeline.feature_matrix = np.load(
        pipeline.root / "checkpoints/feature_matrix.npy"
    )
    dt = time.time() - t0
    n_sigs = len(pipeline.sensitivity)
    fm_shape = pipeline.feature_matrix.shape
    print(
        f"\r  {GREEN}OK{RESET} Signatures loaded: "
        f"{BOLD}{n_sigs}{RESET} rules, "
        f"feature matrix {fm_shape[0]}x{fm_shape[1]} "
        f"{DIM}({elapsed_str(dt)}){RESET}"
    )

    # 3b. Target layer (from J2 metrics)
    j2_path = pipeline.root / "results/metrics/j2_evaluation.json"
    with open(j2_path) as f:
        j2_metrics = json.load(f)
    pipeline.TARGET_LAYER = j2_metrics["train_layer"]

    # 4. GPT-2 Large (the slow one -- downloads ~3 GB first time)
    t0 = time.time()
    print(
        f"  {CYAN}>>{RESET} Loading GPT-2 Large "
        f"{DIM}(downloads ~3 GB on first run)...{RESET}",
        flush=True,
    )
    # Suppress ALL output: Python-level + fd-level (TransformerLens
    # caches stderr at import time, bypassing sys.stderr redirect)
    sys.stdout = _QuietStream()
    sys.stderr = _QuietStream()
    _saved_stdout_fd = os.dup(1)
    _saved_stderr_fd = os.dup(2)
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 1)
    os.dup2(_devnull, 2)
    try:
        pipeline.gpt2 = load_model(device=pipeline.device)
    finally:
        os.dup2(_saved_stdout_fd, 1)
        os.dup2(_saved_stderr_fd, 2)
        os.close(_saved_stdout_fd)
        os.close(_saved_stderr_fd)
        os.close(_devnull)
        sys.stdout = _real_stdout
        sys.stderr = _real_stderr
    dt = time.time() - t0
    print(
        f"\r  {GREEN}OK{RESET} GPT-2 Large loaded: "
        f"{BOLD}36{RESET} layers, d_model={BOLD}1280{RESET}, "
        f"50257 tokens {DIM}({elapsed_str(dt)}){RESET}"
    )

    # 5. Train detectors with two-stage feature selection
    t0 = time.time()
    print(
        f"  {DIM}  Training detection classifiers...{RESET}",
        end="", flush=True,
    )
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.model_selection import train_test_split as tts
    from sklearn.metrics import f1_score as _f1

    labels = np.array(pipeline.dataset.labels)

    # 80/20 stratified split — training on all data overfits
    train_idx, test_idx = tts(
        np.arange(len(labels)), test_size=0.2,
        stratify=labels, random_state=42,
    )
    pipeline.train_idx = train_idx
    pipeline.test_idx = test_idx

    # Stage 1: screen all features to find the important ones
    screening_model = LR(
        random_state=42, max_iter=1000, solver="lbfgs", C=0.01,
    )
    screening_model.fit(
        pipeline.feature_matrix[train_idx], labels[train_idx]
    )
    lr_weights = np.abs(screening_model.coef_[0])
    pipeline.top_feature_indices = np.argsort(lr_weights)[::-1]

    # Stage 2: retrain on top-50 features only (800/50 = 16:1 ratio)
    TOP_K_DETECT = 50
    pipeline._detect_feature_indices = pipeline.top_feature_indices[:TOP_K_DETECT]
    pipeline.sae_detector = LR(
        random_state=42, max_iter=1000, solver="lbfgs", C=0.0001,
    )
    pipeline.sae_detector.fit(
        pipeline.feature_matrix[train_idx][:, pipeline._detect_feature_indices],
        labels[train_idx],
    )
    pipeline.agent_detector = pipeline.sae_detector

    # TF-IDF baseline (train split only)
    train_texts = [pipeline.dataset.texts[i] for i in train_idx]
    train_labels = [pipeline.dataset.labels[i] for i in train_idx]
    lr_pipe, _ = train_tfidf_baseline(train_texts, train_labels, seed=42)
    pipeline.tfidf_detector = lr_pipe

    # Report held-out performance
    test_preds = pipeline.sae_detector.predict(
        pipeline.feature_matrix[test_idx][:, pipeline._detect_feature_indices]
    )
    test_probs = pipeline.sae_detector.predict_proba(
        pipeline.feature_matrix[test_idx][:, pipeline._detect_feature_indices]
    )[:, 1]
    f1_val = _f1(labels[test_idx], test_preds)
    normal_max = test_probs[labels[test_idx] == 0].max()

    dt = time.time() - t0
    print(
        f"\r  {GREEN}OK{RESET} Detectors trained: "
        f"SAE top-{TOP_K_DETECT} (F1={f1_val:.3f}, normal max={normal_max:.3f}) "
        f"+ TF-IDF {DIM}({elapsed_str(dt)}){RESET}"
    )

    # 7. Load results JSONs
    pipeline.results = {}
    metrics_dir = pipeline.root / "results/metrics"
    for p in metrics_dir.glob("*.json"):
        pipeline.results[p.stem] = json.loads(p.read_text(encoding="utf-8"))

    # 8. Phase 2: category fingerprints, steering defense, Phi-3
    t0 = time.time()
    print(
        f"  {DIM}  Loading Phase 2 components...{RESET}",
        end="", flush=True,
    )
    sys.stdout = _QuietStream()
    sys.stderr = _QuietStream()
    try:
        pipeline._load_category_fingerprints()
        pipeline._load_steering_defense()
        pipeline._load_phi3()
    except Exception:
        pass
    sys.stdout = _real_stdout
    sys.stderr = _real_stderr
    dt = time.time() - t0

    phase2_parts = []
    if pipeline.category_fingerprints is not None:
        phase2_parts.append("taxonomy")
    if pipeline.steering_defense is not None:
        phase2_parts.append("steering")
    if pipeline.defense_stack is not None:
        phase2_parts.append("agent+defense")

    if phase2_parts:
        print(
            f"\r  {GREEN}OK{RESET} Phase 2: "
            f"{', '.join(phase2_parts)} {DIM}({elapsed_str(dt)}){RESET}"
        )
    else:
        print(
            f"\r  {YELLOW}--{RESET} Phase 2: detection-only mode "
            f"{DIM}(Phi-3 requires GPU){RESET}"
        )

    pipeline.loaded = True

    return pipeline, build_app


def main():
    total_t0 = time.time()
    root = Path(__file__).parent

    print(BANNER)

    # ── Step 1: Python version ────────────────────────────────────
    step_t0 = time.time()
    step_header(1, 5, "Environment Check")
    check_python()
    info(f"Platform: {sys.platform}")
    step_header(1, 5, "Environment Check", time.time() - step_t0)

    # ── Step 2: Dependencies ──────────────────────────────────────
    step_t0 = time.time()
    step_header(2, 5, "Installing Dependencies")
    install_dependencies(root)

    # Now we can check CUDA properly
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            ok(f"CUDA available: {gpu_name}")
        else:
            info("No GPU detected -- running on CPU (this is fine)")
    except Exception:
        info("Running on CPU")
    step_header(2, 5, "Installing Dependencies", time.time() - step_t0)

    # ── Step 3: Verify checkpoints ────────────────────────────────
    step_t0 = time.time()
    step_header(3, 5, "Verifying Model Artifacts")
    verify_checkpoints(root)
    step_header(3, 5, "Verifying Model Artifacts", time.time() - step_t0)

    # ── Step 4: Load engine ───────────────────────────────────────
    step_t0 = time.time()
    step_header(4, 5, "Initializing Neural IDS Engine")
    pipeline, build_app = load_engine(root)
    step_header(4, 5, "Initializing Neural IDS Engine", time.time() - step_t0)

    # ── Step 5: Launch ────────────────────────────────────────────
    step_header(5, 5, "Launching Dashboard")

    in_colab = "google.colab" in sys.modules
    share = in_colab

    total_dt = time.time() - total_t0
    print()
    print(f"  {GREEN}{BOLD}IRIS Neural IDS ready.{RESET}")
    print(f"  {DIM}Total startup time: {elapsed_str(total_dt)}{RESET}")
    print()

    if not in_colab:
        print(f"  {BOLD}Opening dashboard in your browser...{RESET}")
        print(f"  {DIM}Press Ctrl+C to stop the server{RESET}")
    print()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app = build_app(pipeline)
        app.launch(
            share=share,
            inbrowser=not in_colab,
        )


if __name__ == "__main__":
    main()
