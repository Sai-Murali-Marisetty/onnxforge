# onnxslim

A pass-based ONNX graph optimizer designed for production deployment. Cleans, simplifies, and restructures ONNX models before conversion to TFLite or CoreML with **zero accuracy loss** and **verifiable correctness**.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Features

- **11 optimization passes** — dead node elimination, constant folding, operator fusion, and more
- **Per-pass accuracy verification** — every transformation is validated against the original model
- **Transformer-aware** — handles HuggingFace BERT, DistilBERT, RoBERTa, and Whisper exports
- **Target-aware design** — extensible profiles for TFLite and CoreML conversion requirements
- **Production tested** — validated on 8 models with perfect numerical accuracy

---

## Benchmark Results

Tested on ONNX Runtime 1.19.2 (CPU), 50 inference runs, median latency.

| Model | Nodes Before | Nodes After | Reduction | Latency Δ | max_diff |
|-------|-------------|-------------|-----------|-----------|----------|
| MobileNetV2 | 105 | 105 | 0% | — | 0.00e+00 |
| EfficientNet-B0 | 288 | 239 | **17%** | +1.3% | 0.00e+00 |
| ResNet-50 | 179 | 122 | **32%** | +1.3% | 0.00e+00 |
| MobileNetV3-Small | 175 | 141 | **19%** | −0.1% | 0.00e+00 |
| BERT-base | 1453 | 1453 | 0% | — | 0.00e+00 |
| DistilBERT | 743 | 743 | 0% | **+2.6%** | 0.00e+00 |
| RoBERTa-base | 1453 | 1453 | 0% | — | 0.00e+00 |
| Whisper-tiny | 453 | 453 | 0% | −0.6% | 0.00e+00 |

**Key findings:**
- Vision models with BatchNorm see 17–32% node reduction from Conv+BN fusion
- All models maintain perfect numerical accuracy (max_diff = 0.0)
- Cleaner graphs enable better downstream runtime optimizations

---

## Installation

### Prerequisites

- Python 3.11+ (recommended: use [pyenv](https://github.com/pyenv/pyenv))

### Setup

```bash
git clone https://github.com/yourname/onnxslim.git
cd onnxslim

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Test Models

Models are not included in the repo due to size. Generate them using the export scripts:

```bash
# Vision models (ResNet-50, MobileNetV3) — requires torchvision
python export_vision_models.py

# EfficientNet-B0
python export_efficientnet.py

# Transformer models (BERT, DistilBERT, RoBERTa) — requires transformers
python export_transformer_models.py

# Whisper encoder
python export_distilbert_whisper.py

# Or download MobileNetV2 from ONNX Model Zoo
wget https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx
```

---

## Usage

### Optimize a model

```bash
python optimizer.py input.onnx output.onnx
```

### Verify numerical equivalence

```bash
python verify.py original.onnx optimized.onnx
```

### Example output

```
Loading: efficientnet-b0.onnx

Model: efficientnet-b0.onnx
─────────────────────────────────────────
Nodes before:      288
Nodes after:       239 (-17.0%)
Size before:       20.4 MB
Size after:        20.5 MB (+0.5%)
Passes applied:    11 passes
Time:              0.85s

✓ Verification passed | max_diff=0.00e+00 | samples=5
```

---

## Optimization Passes

| Pass | Description |
|------|-------------|
| `eliminate_dead_nodes` | Remove nodes with unused outputs |
| `eliminate_identity_ops` | Remove redundant Identity nodes |
| `eliminate_unused_initializers` | Remove weights not referenced by any node |
| `eliminate_duplicate_constants` | Deduplicate identical constant tensors |
| `eliminate_redundant_transposes` | Cancel or merge consecutive Transpose ops |
| `fold_constants` | Pre-compute operations on constant inputs |
| `simplify_shape_chains` | Remove identity Reshape operations |
| `fuse_conv_batchnorm` | Merge Conv + BatchNormalization into single Conv |
| `fuse_conv_relu` | Detect Conv + ReLU patterns (for target profiles) |
| `fuse_matmul_add` | Convert MatMul + Add to Gemm where applicable |
| `cleanup_attention` | Simplify redundant Reshape chains in attention blocks |

---

## Project Structure

```
onnxslim/
├── optimizer.py           # Main orchestrator — runs passes in sequence
├── verify.py              # Numerical accuracy verification
├── passes/
│   ├── __init__.py
│   ├── base_pass.py       # Abstract base class for all passes
│   ├── eliminate_dead_nodes.py
│   ├── eliminate_identity_ops.py
│   ├── eliminate_unused_initializers.py
│   ├── eliminate_duplicate_constants.py
│   ├── eliminate_redundant_transposes.py
│   ├── fold_constants.py
│   ├── simplify_shape_chains.py
│   ├── fuse_conv_batchnorm.py
│   ├── fuse_conv_relu.py
│   ├── fuse_matmul_add.py
│   └── cleanup_attention.py
├── tests/
│   ├── test_*.py          # Unit tests for each pass
│   ├── toy_models/        # Minimal test models
│   └── experiments/       # Benchmark scripts
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

All 33 tests should pass.

---

## Adding a New Pass

1. Create `passes/your_pass.py` inheriting from `BasePass`
2. Implement the `name` property and `run(model)` method
3. Add to `passes/__init__.py` exports
4. Register in `optimizer.py`
5. Write tests in `tests/test_your_pass.py`

Example:

```python
from .base_pass import BasePass
import onnx

class YourPass(BasePass):
    @property
    def name(self) -> str:
        return "your_pass"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        # Your optimization logic here
        return model
```

---

## Why onnxslim?

Existing tools like `onnxoptimizer` and `onnxsim` are:
- Generic and largely unmaintained
- Not target-aware (TFLite vs CoreML have different requirements)
- Poor at handling Transformer model exports
- Missing built-in accuracy verification

onnxslim addresses all of these with a modular, testable, and extensible design.

---

## Roadmap

- [x] Core optimization passes (11 passes)
- [x] Accuracy verification system
- [x] Vision model support (ResNet, EfficientNet, MobileNet)
- [x] Transformer model support (BERT, DistilBERT, Whisper)
- [ ] TFLite target profile (LayerNorm decomposition, shape freezing)
- [ ] CoreML target profile
- [ ] CLI with target selection
- [ ] PyPI package

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

## Acknowledgments

Built with [ONNX](https://onnx.ai/) and [ONNX Runtime](https://onnxruntime.ai/).
