# M3 — Eliminate Unused Initializers + Eliminate Duplicate Constants

**Goal:** Implement two passes that target model SIZE rather than node count. These remove dead weight tensors and deduplicate identical constants. First time you'll see meaningful MB reduction — especially on messy exports.

---

## What You're Building

```
onnxslim/
├── optimizer.py                          ← update: register two new passes
├── passes/
│   ├── __init__.py                       ← update: export new passes
│   ├── eliminate_dead_nodes.py           ✅ done
│   ├── eliminate_identity_ops.py         ✅ done
│   ├── eliminate_unused_initializers.py  ← NEW
│   └── eliminate_duplicate_constants.py  ← NEW
└── tests/
    └── test_mobilenetv2.py               ← update: add M3 assertions
```

---

## Background — What Are These?

### Unused Initializers

Initializers are named weight tensors stored in `model.graph.initializer`. They include things like Conv weights, BN scales, embedding tables, biases. When exporters run, they sometimes leave behind initializers that are defined but never actually referenced by any node input. These are dead weight — they bloat file size and do nothing.

**Example scenario:** A model is exported after some layers are pruned or replaced. The old weight tensors remain in `graph.initializer` but no node references them anymore.

```
graph.initializer: [weight_A, weight_B, weight_C]  # C was from a deleted layer
graph.node[0].input: [weight_A]
graph.node[1].input: [weight_B]
# weight_C → never referenced → should be removed
```

### Duplicate Constants

Sometimes the same constant value is stored multiple times under different names. This happens a lot with scalar constants (like epsilon values, scaling factors, positional indices) in HuggingFace exports — the exporter creates a new named constant for each use instead of reusing one.

```
initializer "const_0": [0.00001]   # epsilon for LayerNorm block 1
initializer "const_1": [0.00001]   # epsilon for LayerNorm block 2  ← identical
initializer "const_2": [0.00001]   # epsilon for LayerNorm block 3  ← identical
```

All three can be collapsed to one. Every node that referenced `const_1` or `const_2` now points to `const_0` instead.

---

## Pass 1: `eliminate_unused_initializers.py`

### Algorithm

1. Collect all initializer names from `model.graph.initializer`
2. Collect all input names actually used by any node across the entire graph
3. Also collect graph input names (these are runtime inputs, not weights — don't touch them)
4. Remove any initializer whose name appears in neither node inputs nor graph inputs

### Important edge case: graph inputs vs initializers

In ONNX, `model.graph.input` contains BOTH runtime inputs (like the image tensor) AND initializers (weights). Some older models list their weights in both places. You need to be careful not to remove initializers that are also declared as graph inputs — that would break the model structure.

Safe rule: only remove an initializer if it is not referenced by any node AND not listed in `graph.input`.

### Code

```python
import onnx
from passes.base_pass import BasePass

class EliminateUnusedInitializers(BasePass):

    @property
    def name(self) -> str:
        return "eliminate_unused_initializers"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        # All names used as inputs across all nodes
        used_inputs = set()
        for node in model.graph.node:
            for inp in node.input:
                if inp:  # skip empty strings (optional inputs)
                    used_inputs.add(inp)

        # Names declared as graph inputs (runtime inputs + legacy weight declarations)
        graph_input_names = {inp.name for inp in model.graph.input}

        # Keep initializers that are actually used
        kept = []
        removed = 0
        for initializer in model.graph.initializer:
            if initializer.name in used_inputs or initializer.name in graph_input_names:
                kept.append(initializer)
            else:
                removed += 1

        # Rebuild initializer list in-place
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)

        if removed > 0:
            print(f"    → removed {removed} unused initializer(s)")

        return model
```

---

## Pass 2: `eliminate_duplicate_constants.py`

### Algorithm

1. For each initializer, compute a hash of its raw data bytes + dtype + shape
2. Group initializers by hash
3. For each group with more than one member, pick the first as the canonical one
4. Replace all references to the duplicates with the canonical name in every node input
5. Also update graph outputs if any reference the removed names (rare but possible)
6. Remove the duplicate initializers from the graph

### Hashing approach

```python
import hashlib
import numpy as np
from onnx import numpy_helper

def _hash_initializer(tensor) -> str:
    """Hash an initializer by its dtype, shape, and raw data."""
    arr = numpy_helper.to_array(tensor)
    # Include shape and dtype in hash to avoid collisions across different-shaped arrays
    meta = f"{arr.dtype}:{arr.shape}"
    data_hash = hashlib.md5(arr.tobytes()).hexdigest()
    return f"{meta}:{data_hash}"
```

### Code

```python
import onnx
import hashlib
from onnx import numpy_helper
from passes.base_pass import BasePass

def _hash_initializer(tensor) -> str:
    arr = numpy_helper.to_array(tensor)
    meta = f"{arr.dtype}:{arr.shape}"
    data_hash = hashlib.md5(arr.tobytes()).hexdigest()
    return f"{meta}:{data_hash}"

class EliminateDuplicateConstants(BasePass):

    @property
    def name(self) -> str:
        return "eliminate_duplicate_constants"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        # Group initializers by content hash
        hash_to_canonical = {}   # hash → first initializer name seen
        remap = {}               # duplicate name → canonical name

        for initializer in model.graph.initializer:
            h = _hash_initializer(initializer)
            if h not in hash_to_canonical:
                hash_to_canonical[h] = initializer.name
            else:
                # This initializer is a duplicate — map it to the canonical one
                canonical = hash_to_canonical[h]
                if initializer.name != canonical:
                    remap[initializer.name] = canonical

        if not remap:
            return model  # nothing to do

        # Rewrite all node inputs
        for node in model.graph.node:
            for i, inp in enumerate(node.input):
                if inp in remap:
                    node.input[i] = remap[inp]

        # Rewrite graph outputs (rare but correct)
        for output in model.graph.output:
            if output.name in remap:
                output.name = remap[output.name]

        # Remove duplicate initializers
        kept = [init for init in model.graph.initializer
                if init.name not in remap]

        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)

        print(f"    → removed {len(remap)} duplicate constant(s)")

        return model
```

---

## Update `passes/__init__.py`

```python
from passes.eliminate_dead_nodes import EliminateDeadNodes
from passes.eliminate_identity_ops import EliminateIdentityOps
from passes.eliminate_unused_initializers import EliminateUnusedInitializers
from passes.eliminate_duplicate_constants import EliminateDuplicateConstants
```

---

## Update `optimizer.py` — Register Passes

Import and register in order. Unused initializers should run before duplicate constants — no point deduplicating something that's already going to be removed.

```python
from passes.eliminate_dead_nodes import EliminateDeadNodes
from passes.eliminate_identity_ops import EliminateIdentityOps
from passes.eliminate_unused_initializers import EliminateUnusedInitializers
from passes.eliminate_duplicate_constants import EliminateDuplicateConstants

registered_passes = [
    EliminateDeadNodes(),
    EliminateIdentityOps(),
    EliminateUnusedInitializers(),
    EliminateDuplicateConstants(),
]
```

---

## Update `tests/test_mobilenetv2.py`

Add assertions for M3 passes. MobileNetV2 is clean so reductions will be zero — that's fine and expected. The test just confirms the passes don't break anything.

```python
def test_m3_passes(model_path="mobilenetv2-12.onnx", output_path="mobilenetv2-12-m3.onnx"):
    from passes.eliminate_unused_initializers import EliminateUnusedInitializers
    from passes.eliminate_duplicate_constants import EliminateDuplicateConstants
    import onnx
    from verify import verify

    original = onnx.load(model_path)
    model = onnx.load(model_path)

    model = EliminateUnusedInitializers().run(model)
    model = EliminateDuplicateConstants().run(model)

    report = verify(original, model, n_samples=10)
    assert report.passed
    assert report.max_diff < 1e-5

    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    init_before = len(original.graph.initializer)
    init_after  = len(model.graph.initializer)
    print(f"✓ M3 test passed")
    print(f"  Initializers: {init_before} → {init_after}")
    print(f"  Max diff: {report.max_diff:.2e}")

if __name__ == "__main__":
    test_m3_passes()
```

---

## Smoke Test

```bash
# Run full optimizer (all 4 passes now)
python optimizer.py mobilenetv2-12.onnx mobilenetv2-12-m3.onnx

# Verify
python verify.py mobilenetv2-12.onnx mobilenetv2-12-m3.onnx

# Run test file
python tests/test_mobilenetv2.py
```

**Expected output on MobileNetV2** (clean model — minimal reduction):
```
Loading: mobilenetv2-12.onnx
  Running pass: eliminate_dead_nodes ... ✓
  Running pass: eliminate_identity_ops ... ✓
  Running pass: eliminate_unused_initializers ... ✓
  Running pass: eliminate_duplicate_constants ... ✓

Model: mobilenetv2-12.onnx
─────────────────────────────────────────
Nodes before:      105
Nodes after:       105 (-0.0%)
Size before:       13.32 MB
Size after:        13.32 MB (-0.0%)
Passes applied:    eliminate_dead_nodes, eliminate_identity_ops,
                   eliminate_unused_initializers, eliminate_duplicate_constants
Time:              ~0.3s
```

Zero reduction on MobileNetV2 is still a pass. These passes will show real numbers on BERT and Whisper.

---

## Definition of Done

- [ ] `eliminate_unused_initializers.py` implemented and importable
- [ ] `eliminate_duplicate_constants.py` implemented and importable
- [ ] `passes/__init__.py` updated with both new passes
- [ ] `optimizer.py` updated — all 4 passes registered in correct order
- [ ] `tests/test_mobilenetv2.py` updated with M3 test function
- [ ] `python optimizer.py mobilenetv2-12.onnx mobilenetv2-12-m3.onnx` runs clean
- [ ] `verify.py` confirms zero accuracy diff on 10 samples
- [ ] `onnx.checker.check_model()` passes on output model

---

## Known Gotchas

**Empty node inputs** — ONNX uses empty strings `""` as placeholders for optional node inputs (e.g. missing bias). Always guard with `if inp:` before adding to your used set, or you'll end up matching against empty-string initializers that don't exist.

**MD5 collision risk** — negligible for this use case but worth knowing. If you're paranoid, swap `hashlib.md5` for `hashlib.sha256`. The performance difference is irrelevant at model-init time.

**numpy_helper.to_array on large tensors** — for big embedding tables (BERT has a 30k vocab × 768 embedding = ~90MB), this materializes the full array just to hash it. Fine for now. If it gets slow in M9+, switch to hashing `tensor.raw_data` directly without converting to numpy.

**Pass order matters** — always run `eliminate_unused_initializers` before `eliminate_duplicate_constants`. If you run duplicates first, you may canonicalize a name that then gets removed as unused — harmless but wasteful.

---

## Next: M4

Once M3 is green → `micro_plan_M4.md` — implement `eliminate_redundant_transposes`. This is the first **high-impact** structural pass. YOLOv8n will show real node count reduction here — PyTorch exports NCHW, TFLite wants NHWC, and the exporter often inserts cancelling Transpose pairs that serve no purpose.
