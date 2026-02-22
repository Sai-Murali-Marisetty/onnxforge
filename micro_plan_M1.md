# M1 — Project Scaffold + Base Pass + Verify

**Goal:** Get a working skeleton where you can run the optimizer end-to-end with zero real passes. Nothing smart happens yet — but the plumbing works and verify.py confirms it.

---

## What You're Building

```
onnxslim/
├── optimizer.py        ← orchestrator: loads model, runs passes in sequence, saves output
├── verify.py           ← runs inputs through original + optimized, compares outputs
├── passes/
│   ├── __init__.py
│   └── base_pass.py    ← abstract class all passes will inherit from
├── requirements.txt
└── README.md
```

Nothing in `passes/` does real work yet. The point is: `python optimizer.py model.onnx` runs cleanly, verify passes, and you have a foundation to hang everything else on.

---

## Step-by-Step Tasks

### 1. Environment Setup

#### Install pyenv (if not already installed)

pyenv manages the Python version so the project is reproducible across machines and environments (local M3, Colab, Azure later).

**macOS:**
```bash
brew install pyenv
```

Add to your shell config (`~/.zshrc` or `~/.bash_profile`):
```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Then reload:
```bash
source ~/.zshrc
```

#### Pin Python version and create venv

```bash
mkdir onnxslim && cd onnxslim

# Install and pin Python 3.11.9 for this project
pyenv install 3.11.9
pyenv local 3.11.9        # creates .python-version file — commit this

# Confirm version
python --version           # should print Python 3.11.9

# Create virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install onnx onnxruntime numpy
```

#### `.python-version` file (auto-created by pyenv local)
```
3.11.9
```
Commit this file. It tells anyone (or any machine) exactly which Python version to use.

#### `.gitignore`

Create this before your first commit:
```
venv/
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.DS_Store
*.onnx          # models are large — don't commit them
models/
```

#### `requirements.txt`
```
onnx>=1.14.0
onnxruntime>=1.16.0
numpy>=1.24.0
```

---

### 2. README.md

Create this at the root of the project. It should be the single source of truth for setup and usage.

````markdown
# onnxslim

A pass-based ONNX graph optimizer. Cleans, simplifies, and restructures ONNX models
before conversion to TFLite or CoreML. Zero accuracy loss. Verifiable. Target-aware.

Built as a transparent alternative to onnxsim/onnxoptimizer — with per-pass accuracy
verification and Transformer-specific optimization.

---

## Why

- `onnxoptimizer` (Meta) and `onnxsim` are generic and largely unmaintained
- Neither is target-aware (TFLite vs CoreML have different failure patterns)
- Neither handles HuggingFace Transformer exports well
- Neither verifies accuracy preservation with sample inputs

onnxslim does all of this.

---

## Project Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1 | Scaffold + base pass + verify | 🔧 In progress |
| M2 | eliminate_dead_nodes + eliminate_identity_ops | ⬜ Planned |
| M3 | eliminate_unused_initializers + eliminate_duplicate_constants | ⬜ Planned |
| M4 | eliminate_redundant_transposes | ⬜ Planned |
| M5 | fold_constants | ⬜ Planned |
| M6 | simplify_shape_chains | ⬜ Planned |
| M7 | fuse_conv_batchnorm | ⬜ Planned |
| M8 | fuse_conv_relu + fuse_matmul_add | ⬜ Planned |
| M9 | cleanup_attention (BERT) | ⬜ Planned |
| M10 | cleanup_attention (Whisper) | ⬜ Planned |
| M11 | TFLite target profile | ⬜ Planned |
| M12 | CoreML target profile | ⬜ Planned |
| M13 | Benchmark report | ⬜ Planned |

---

## Setup

### Prerequisites

- [pyenv](https://github.com/pyenv/pyenv) — manages Python version

### Install

```bash
# Clone the repo
git clone https://github.com/yourname/onnxslim.git
cd onnxslim

# pyenv will auto-read .python-version and use Python 3.11.9
pyenv install 3.11.9      # skip if already installed
python --version           # confirm 3.11.9

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the optimizer

```bash
python optimizer.py input.onnx output.onnx
```

### Verify two models are numerically equivalent

```bash
python verify.py original.onnx optimized.onnx
```

### Example output

```
Loading: mobilenetv2-12.onnx

Model: mobilenetv2-12.onnx
─────────────────────────────────────────
Nodes before:      352
Nodes after:       298 (-15.3%)
Size before:       13.3 MB
Size after:        12.1 MB (-9.0%)
Passes applied:    dead_nodes, identity, transpose, const_fold, conv_bn_fuse
Time:              1.2s

✓ Verification passed | max_diff=3.00e-06 | samples=5
```

---

## Project Structure

```
onnxslim/
├── optimizer.py              # Orchestrator — runs passes in sequence
├── verify.py                 # Accuracy verification
├── passes/
│   ├── base_pass.py          # Abstract base class for all passes
│   ├── eliminate_dead_nodes.py
│   ├── eliminate_identity_ops.py
│   └── ...                   # One file per pass
├── targets/
│   ├── tflite_profile.py     # TFLite-specific pre-fixes
│   └── coreml_profile.py     # CoreML-specific pre-fixes
├── tests/
│   └── ...
├── benchmarks/
│   └── size_latency_before_after.py
├── .python-version           # Pinned to 3.11.9 via pyenv
├── requirements.txt
└── README.md
```

---

## Dev Environment

| Machine | Role |
|---------|------|
| MacBook M3 Pro | Primary dev, CoreML conversion |
| Colab Pro | Heavy quantization, batch testing |
| Azure (Phase 3) | Hosted demo endpoint |

---

## License

MIT
````

---

### 2. `passes/base_pass.py`

This is the contract every pass must fulfill.

```python
from abc import ABC, abstractmethod
import onnx

class BasePass(ABC):
    """
    Abstract base class for all optimization passes.
    Every pass receives an onnx.ModelProto and returns a modified one.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging and reports."""
        pass

    @abstractmethod
    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """
        Apply the optimization pass.
        Must return a valid ONNX model.
        Must not alter accuracy.
        """
        pass

    def __repr__(self):
        return f"<Pass: {self.name}>"
```

Create `passes/__init__.py` as empty file.

---

### 3. `verify.py`

Runs N random inputs through original and optimized models. Compares outputs. Raises if drift exceeds tolerance.

```python
import numpy as np
import onnxruntime as ort
import onnx
from dataclasses import dataclass
from typing import Optional

TOLERANCE = 1e-5  # max acceptable absolute difference

@dataclass
class VerificationReport:
    passed: bool
    max_diff: float
    samples: int
    failed_at_sample: Optional[int] = None

class AccuracyLossError(Exception):
    pass

def _get_input_specs(model: onnx.ModelProto):
    """Extract input names and shapes from model graph."""
    inputs = []
    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            # Use 1 for dynamic/unknown dims
            shape.append(dim.dim_value if dim.dim_value > 0 else 1)
        dtype = inp.type.tensor_type.elem_type
        inputs.append((inp.name, shape, dtype))
    return inputs

def _generate_random_input(input_specs):
    """Generate random numpy arrays matching model input specs."""
    feed = {}
    for name, shape, dtype in input_specs:
        if dtype == 1:   # FLOAT
            feed[name] = np.random.randn(*shape).astype(np.float32)
        elif dtype == 7: # INT64
            feed[name] = np.random.randint(0, 128, shape).astype(np.int64)
        elif dtype == 6: # INT32
            feed[name] = np.random.randint(0, 128, shape).astype(np.int32)
        else:
            feed[name] = np.random.randn(*shape).astype(np.float32)
    return feed

def verify(
    original_model: onnx.ModelProto,
    optimized_model: onnx.ModelProto,
    n_samples: int = 5,
    tolerance: float = TOLERANCE
) -> VerificationReport:
    """
    Run n_samples random inputs through both models.
    Compare all outputs. Raise AccuracyLossError if diff exceeds tolerance.
    """
    sess_orig = ort.InferenceSession(original_model.SerializeToString())
    sess_opt  = ort.InferenceSession(optimized_model.SerializeToString())

    input_specs = _get_input_specs(original_model)
    output_names = [o.name for o in original_model.graph.output]

    global_max_diff = 0.0

    for i in range(n_samples):
        feed = _generate_random_input(input_specs)

        out_orig = sess_orig.run(output_names, feed)
        out_opt  = sess_opt.run(output_names, feed)

        for orig, opt in zip(out_orig, out_opt):
            diff = np.max(np.abs(np.array(orig) - np.array(opt)))
            global_max_diff = max(global_max_diff, diff)

            if diff > tolerance:
                raise AccuracyLossError(
                    f"Sample {i}: max diff {diff:.2e} exceeds tolerance {tolerance:.2e}"
                )

    return VerificationReport(passed=True, max_diff=global_max_diff, samples=n_samples)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python verify.py original.onnx optimized.onnx")
        sys.exit(1)

    orig = onnx.load(sys.argv[1])
    opt  = onnx.load(sys.argv[2])
    report = verify(orig, opt)
    print(f"✓ Verification passed | max_diff={report.max_diff:.2e} | samples={report.samples}")
```

---

### 4. `optimizer.py`

Orchestrator. Loads model, runs all registered passes in order, verifies after each pass, saves output.

```python
import onnx
import time
from typing import List
from passes.base_pass import BasePass
from verify import verify, AccuracyLossError

def count_nodes(model: onnx.ModelProto) -> int:
    return len(model.graph.node)

def model_size_mb(model: onnx.ModelProto) -> float:
    return model.ByteSize() / (1024 * 1024)

def optimize(
    model_path: str,
    output_path: str,
    passes: List[BasePass],
    verify_each_pass: bool = True,
    n_verify_samples: int = 5
) -> dict:
    """
    Load model, run passes in sequence, verify, save.
    Returns a report dict with before/after stats.
    """
    print(f"\nLoading: {model_path}")
    original = onnx.load(model_path)
    model = onnx.load(model_path)  # working copy

    nodes_before = count_nodes(model)
    size_before  = model_size_mb(model)
    passes_applied = []

    start = time.time()

    for p in passes:
        print(f"  Running pass: {p.name} ...", end=" ")
        model = p.run(model)

        if verify_each_pass:
            try:
                verify(original, model, n_samples=n_verify_samples)
                print("✓")
            except AccuracyLossError as e:
                print(f"✗ FAILED — {e}")
                raise
        else:
            print("(verify skipped)")

        passes_applied.append(p.name)

    elapsed = time.time() - start
    nodes_after = count_nodes(model)
    size_after  = model_size_mb(model)

    onnx.save(model, output_path)

    report = {
        "nodes_before":   nodes_before,
        "nodes_after":    nodes_after,
        "nodes_delta_pct": round((nodes_before - nodes_after) / max(nodes_before, 1) * 100, 1),
        "size_before_mb": round(size_before, 2),
        "size_after_mb":  round(size_after, 2),
        "size_delta_pct": round((size_before - size_after) / max(size_before, 1e-9) * 100, 1),
        "passes_applied": passes_applied,
        "time_sec":       round(elapsed, 2),
    }

    _print_report(model_path, report)
    return report

def _print_report(model_path: str, r: dict):
    print(f"""
Model: {model_path}
─────────────────────────────────────────
Nodes before:      {r['nodes_before']}
Nodes after:       {r['nodes_after']} ({-r['nodes_delta_pct']:+.1f}%)
Size before:       {r['size_before_mb']} MB
Size after:        {r['size_after_mb']} MB ({-r['size_delta_pct']:+.1f}%)
Passes applied:    {', '.join(r['passes_applied']) if r['passes_applied'] else 'none'}
Time:              {r['time_sec']}s
""")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python optimizer.py input.onnx output.onnx")
        sys.exit(1)

    # M1: no real passes yet — just prove the pipeline runs
    registered_passes = []

    optimize(
        model_path=sys.argv[1],
        output_path=sys.argv[2],
        passes=registered_passes,
    )
```

---

### 5. Smoke Test — MobileNetV2

Download a small ONNX model and run the pipeline:

```bash
# Download MobileNetV2 from ONNX Model Zoo
wget https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx

# Run optimizer (zero passes — just prove the scaffold works)
python optimizer.py mobilenetv2-12.onnx mobilenetv2-12-opt.onnx

# Run verify standalone
python verify.py mobilenetv2-12.onnx mobilenetv2-12-opt.onnx
```

**Expected output:**
```
Loading: mobilenetv2-12.onnx

Model: mobilenetv2-12.onnx
─────────────────────────────────────────
Nodes before:      352
Nodes after:       352 (+0.0%)
Size before:       13.3 MB
Size after:        13.3 MB (+0.0%)
Passes applied:    none
Time:              0.3s

✓ Verification passed | max_diff=0.00e+00 | samples=5
```

---

## Definition of Done

- [ ] pyenv installed and `python --version` confirms 3.11.9
- [ ] `.python-version` file committed
- [ ] `.gitignore` in place (venv, pycache, *.onnx excluded)
- [ ] `requirements.txt` created and `pip install -r requirements.txt` runs clean
- [ ] `README.md` written at repo root
- [ ] Project structure created, venv working
- [ ] `base_pass.py` defines clean abstract interface
- [ ] `verify.py` runs standalone and compares two models correctly
- [ ] `optimizer.py` runs with zero passes on MobileNetV2 without errors
- [ ] Verify confirms zero diff when no passes are applied (sanity check)
- [ ] Report prints correctly

---

## Known Gotchas

**Dynamic shapes in verify.py** — MobileNetV2 has a fixed input shape so M1 is safe. When you hit BERT in M6, inputs will have dynamic sequence length. You'll update `_get_input_specs` then to handle named dynamic dims (e.g. `batch_size`, `sequence_length`).

**Model Zoo URL** — The ONNX model zoo URLs change occasionally. If the wget fails, grab the model from [https://github.com/onnx/models](https://github.com/onnx/models) directly.

**ByteSize() on large models** — accurate enough for reporting but not byte-perfect. Fine for M1.

---

## Next: M2

Once this passes cleanly → `micro_plan_M2.md` — implement `eliminate_dead_nodes` + `eliminate_identity_ops` and watch the node count actually drop.
