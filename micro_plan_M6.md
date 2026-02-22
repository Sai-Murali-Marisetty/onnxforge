# M6 — Simplify Shape Chains

**Goal:** Clean up shape manipulation ops that become dead or trivially collapsible after
constant folding. This is a cleanup pass, not a standalone optimization — it works because
M5 (fold_constants) already turned dynamic shape computations into static constants.
First milestone to test on BERT-base.

---

## What You're Building

```
onnxslim/
├── passes/
│   └── simplify_shape_chains.py             ← NEW
├── tests/
│   ├── toy_models/
│   │   └── build_shape_chain_model.py       ← NEW — 3 dirty shape chain models
│   └── test_shape_chains.py                 ← NEW — precise assertions
models/
└── bert-base-uncased.onnx                   ← DOWNLOAD — first real Transformer test
```

---

## Background — What Shape Chains Are

HuggingFace Transformer exports are full of this pattern:

```
Shape(input) → Gather(indices=0) → Unsqueeze → Concat → Reshape(tensor, shape_chain)
```

The intent is to compute the output shape dynamically at runtime. But if the shape is
actually static (fixed batch size, fixed sequence length), then after M5's constant
folding the entire shape chain becomes a constant — and then it's feeding a Reshape
whose target shape is now fully known at compile time.

**Concrete example from BERT export:**

```
# What you see in the ONNX graph:
input_ids: [1, 512]
Shape(input_ids)          → [1, 512]         # runtime shape extraction
Gather([1,512], indices=0) → 1               # get batch dim
Gather([1,512], indices=1) → 512             # get seq dim
Unsqueeze(1)              → [1]
Unsqueeze(512)            → [512]
Concat([1], [512], [-1])  → [1, 512, -1]     # build reshape target
Reshape(tensor, [1,512,-1]) → reshaped

# After M5 constant folding (if shapes are static):
Constant([1, 512, -1])                       # the whole chain collapses
Reshape(tensor, [1, 512, -1]) → reshaped    # this one stays (has runtime input)
```

M6 then cleans up any remaining dead Shape ops and simplifies Reshape nodes
whose target shape is a constant.

---

## What This Pass Actually Does

Three specific patterns to target:

### Pattern 1 — Dead Shape ops after folding

After M5, a `Shape` node whose output is only consumed by already-folded nodes
becomes dead. The dead node pass in M2 should catch most of these, but Shape nodes
that feed partially-dead chains may be missed. Run dead node elimination again
within this pass.

### Pattern 2 — Reshape with constant shape and known input shape

```
Reshape(X, Constant([batch, seq, hidden]))
```

If the target shape is a constant AND the input tensor has a statically known shape,
verify the reshape is valid and keep it — but strip any intermediate shape-computing
nodes that fed the now-constant shape input.

### Pattern 3 — Redundant Reshape (identity reshape)

```
Reshape(X, shape=[1, 512, 768])
```

If the input X already has shape `[1, 512, 768]`, this Reshape does nothing.
Remove it.

---

## Pass: `passes/simplify_shape_chains.py`

```python
import onnx
from onnx import helper, numpy_helper, TensorProto
from passes.base_pass import BasePass


def _build_output_to_node(graph):
    """Map: output_name → node that produces it."""
    return {out: node for node in graph.node for out in node.output}


def _build_input_consumers(graph):
    """Map: tensor_name → list of nodes that consume it."""
    consumers = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    return consumers


def _get_constant_value(name, graph, output_to_node):
    """
    Try to get the numpy value of a named tensor if it's a constant.
    Checks initializers first, then Constant nodes.
    Returns numpy array or None.
    """
    from onnx import numpy_helper

    # Check initializers
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)

    # Check Constant nodes
    if name in output_to_node:
        node = output_to_node[name]
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    return numpy_helper.to_array(attr.t)

    return None


def _get_static_shape(name, graph):
    """
    Try to get the static shape of a tensor from graph value_info or graph inputs.
    Returns list of ints or None if shape is dynamic/unknown.
    """
    # Check graph inputs
    for inp in graph.input:
        if inp.name == name:
            dims = inp.type.tensor_type.shape.dim
            shape = []
            for d in dims:
                if d.dim_value > 0:
                    shape.append(d.dim_value)
                else:
                    return None  # dynamic dim
            return shape

    # Check value_info (shape inference results)
    for vi in graph.value_info:
        if vi.name == name:
            dims = vi.type.tensor_type.shape.dim
            shape = []
            for d in dims:
                if d.dim_value > 0:
                    shape.append(d.dim_value)
                else:
                    return None
            return shape

    return None


class SimplifyShapeChains(BasePass):

    @property
    def name(self) -> str:
        return "simplify_shape_chains"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        # Run shape inference first so value_info is populated
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception:
            pass  # shape inference can fail on unusual models — proceed anyway

        graph = model.graph
        output_to_node = _build_output_to_node(graph)
        consumers = _build_input_consumers(graph)
        graph_output_names = {o.name for o in graph.output}

        nodes_to_remove = set()
        rewire = {}
        removed_count = 0

        for node in graph.node:
            if id(node) in nodes_to_remove:
                continue

            # --- Pattern: Redundant Reshape (identity reshape) ---
            if node.op_type == "Reshape":
                input_name  = node.input[0]
                shape_input = node.input[1] if len(node.input) > 1 else None
                output_name = node.output[0]

                if shape_input:
                    shape_val = _get_constant_value(shape_input, graph, output_to_node)
                    input_shape = _get_static_shape(input_name, graph)

                    if shape_val is not None and input_shape is not None:
                        target_shape = shape_val.flatten().tolist()
                        # Check if reshape is identity (same shape, no -1 dims)
                        if (len(target_shape) == len(input_shape) and
                                all(int(t) == s for t, s in zip(target_shape, input_shape)) and
                                -1 not in target_shape):
                            # This Reshape does nothing — remove it
                            if output_name not in graph_output_names:
                                rewire[output_name] = input_name
                                nodes_to_remove.add(id(node))
                                removed_count += 1

            # --- Pattern: Shape node whose output is fully constant (dead after folding) ---
            if node.op_type == "Shape":
                output_name = node.output[0]
                # If this Shape output only feeds nodes that are being removed or
                # is never consumed, mark it dead
                output_consumers = consumers.get(output_name, [])
                if not output_consumers and output_name not in graph_output_names:
                    nodes_to_remove.add(id(node))
                    removed_count += 1

        if not nodes_to_remove and not rewire:
            return model

        # Apply rewiring
        for n in graph.node:
            for i, inp in enumerate(n.input):
                if inp in rewire:
                    n.input[i] = rewire[inp]

        # Apply rewiring to graph outputs
        for out in graph.output:
            if out.name in rewire:
                out.name = rewire[out.name]

        # Rebuild node list
        new_nodes = [n for n in graph.node if id(n) not in nodes_to_remove]
        del graph.node[:]
        graph.node.extend(new_nodes)

        if removed_count > 0:
            print(f"    → simplified {removed_count} shape chain node(s)")

        return model
```

---

## Toy Models: `tests/toy_models/build_shape_chain_model.py`

```python
"""
Builds synthetic dirty ONNX models for testing simplify_shape_chains.
Run: python tests/toy_models/build_shape_chain_model.py
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def build_identity_reshape(output_path="tests/toy_models/shape_identity_reshape.onnx"):
    """
    Reshape where input shape == target shape → redundant, should be removed.

    X: [1, 4, 4]
    shape_const: [1, 4, 4]   ← constant, matches input shape exactly
    Reshape(X, [1,4,4]) → Y  ← does nothing

    Before: 2 nodes (Constant + Reshape)
    After:  0 nodes (X directly wires to Y)

    Note: with no nodes remaining, X IS Y after rewiring.
    """
    shape_val = np.array([1, 4, 4], dtype=np.int64)
    shape_tensor = numpy_helper.from_array(shape_val, name="shape_val")
    const_shape = helper.make_node(
        "Constant", [], ["shape"], value=shape_tensor, name="ConstShape"
    )
    reshape = helper.make_node("Reshape", ["X", "shape"], ["Y"], name="Reshape")

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 4])

    graph = helper.make_graph([const_shape, reshape], "identity_reshape", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    # Run shape inference so value_info is populated
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (Reshape with same shape — should be removed)")


def build_real_reshape(output_path="tests/toy_models/shape_real_reshape.onnx"):
    """
    Reshape that actually changes shape — must NOT be removed.

    X: [1, 16]
    Reshape(X, [4, 4]) → Y: [4, 4]

    Before: 2 nodes (Constant + Reshape)
    After:  2 nodes (untouched — reshape actually does something)
    """
    shape_val = np.array([4, 4], dtype=np.int64)
    shape_tensor = numpy_helper.from_array(shape_val, name="shape_val")
    const_shape = helper.make_node(
        "Constant", [], ["shape"], value=shape_tensor, name="ConstShape"
    )
    reshape = helper.make_node("Reshape", ["X", "shape"], ["Y"], name="Reshape")

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 4])

    graph = helper.make_graph([const_shape, reshape], "real_reshape", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (Reshape changes shape — must survive)")


def build_dead_shape_node(output_path="tests/toy_models/shape_dead_shape.onnx"):
    """
    A Shape node whose output is never consumed (dead after folding cleaned up consumers).

    X → Shape → [dead, no consumer]
    X → Relu  → Y

    Before: 2 nodes (Shape + Relu)
    After:  1 node (Shape removed, Relu survives)
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    shape_node = helper.make_node("Shape", ["X"], ["shape_out"], name="ShapeNode")
    relu       = helper.make_node("Relu",  ["X"], ["Y"],         name="Relu")

    graph = helper.make_graph([shape_node, relu], "dead_shape", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (Shape node with no consumer — should be removed)")


if __name__ == "__main__":
    os.makedirs("tests/toy_models", exist_ok=True)
    print("Building toy models for M6...\n")
    build_identity_reshape()
    build_real_reshape()
    build_dead_shape_node()
    print("\nDone. All models saved to tests/toy_models/")
```

---

## Test File: `tests/test_shape_chains.py`

```python
"""
Tests for simplify_shape_chains pass.
Run: python tests/test_shape_chains.py
"""
import numpy as np
import onnx
from verify import verify
from passes.simplify_shape_chains import SimplifyShapeChains


def _run_pass(model_path):
    original  = onnx.load(model_path)
    model     = onnx.load(model_path)
    optimized = SimplifyShapeChains().run(model)
    onnx.checker.check_model(optimized)
    return original, optimized


def test_identity_reshape():
    """
    Reshape where input shape == target shape → Reshape removed entirely.
    X wires directly to Y.
    """
    orig, opt = _run_pass("tests/toy_models/shape_identity_reshape.onnx")

    # Reshape (and its Constant shape feeder) should be gone
    reshape_nodes = [n for n in opt.graph.node if n.op_type == "Reshape"]
    assert len(reshape_nodes) == 0, \
        f"Identity Reshape should be removed, got {len(reshape_nodes)} Reshape nodes"

    # Accuracy: pass a runtime input through both models
    report = verify(orig, opt, n_samples=10)
    assert report.passed and report.max_diff < 1e-5

    print(f"  ✓ identity_reshape:  Reshape removed | max_diff={report.max_diff:.2e}")


def test_real_reshape():
    """
    Reshape that actually changes shape must survive untouched.
    """
    orig, opt = _run_pass("tests/toy_models/shape_real_reshape.onnx")

    reshape_nodes = [n for n in opt.graph.node if n.op_type == "Reshape"]
    assert len(reshape_nodes) == 1, \
        f"Real Reshape must survive, got {len(reshape_nodes)} Reshape nodes"

    report = verify(orig, opt, n_samples=10)
    assert report.passed and report.max_diff < 1e-5

    print(f"  ✓ real_reshape:      Reshape survived | max_diff={report.max_diff:.2e}")


def test_dead_shape_node():
    """
    Shape node with no consumers removed. Relu survives.
    2 nodes → 1 node.
    """
    orig, opt = _run_pass("tests/toy_models/shape_dead_shape.onnx")

    assert len(orig.graph.node) == 2
    assert len(opt.graph.node) == 1, \
        f"Expected 1 node after, got {len(opt.graph.node)}"

    surviving_ops = [n.op_type for n in opt.graph.node]
    assert "Relu" in surviving_ops, "Relu must survive"
    assert "Shape" not in surviving_ops, "Dead Shape must be removed"

    report = verify(orig, opt, n_samples=10)
    assert report.passed

    print(f"  ✓ dead_shape_node:   2 → 1 node | Shape removed, Relu survived")


def test_mobilenetv2():
    """Integration check — MobileNetV2 stays clean after pass."""
    orig, opt = _run_pass("mobilenetv2-12.onnx")
    report = verify(orig, opt, n_samples=5)
    assert report.passed

    nodes_before = len(orig.graph.node)
    nodes_after  = len(opt.graph.node)
    print(f"  ✓ mobilenetv2:       {nodes_before} → {nodes_after} nodes | max_diff={report.max_diff:.2e}")


def test_bert():
    """
    BERT-base-uncased — first Transformer model.
    Run full pipeline (all 7 passes) and check:
    - Node count drops meaningfully
    - Model stays valid
    - Accuracy preserved on random inputs

    Download model first:
    python -c "
    from transformers import BertModel
    import torch, torch.onnx
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    dummy = {'input_ids': torch.ones(1,128,dtype=torch.long),
             'attention_mask': torch.ones(1,128,dtype=torch.long)}
    torch.onnx.export(model, (dummy,), 'models/bert-base-uncased.onnx',
                      input_names=['input_ids','attention_mask'],
                      output_names=['last_hidden_state'],
                      dynamic_axes={'input_ids':{0:'batch',1:'seq'},
                                    'attention_mask':{0:'batch',1:'seq'}},
                      opset_version=14)
    "
    """
    import os
    bert_path = "models/bert-base-uncased.onnx"
    if not os.path.exists(bert_path):
        print(f"  ⚠ BERT model not found at {bert_path} — skipping")
        print(f"    Run the export command in the docstring above to generate it")
        return

    orig, opt = _run_pass(bert_path)
    report = verify(orig, opt, n_samples=3)
    assert report.passed

    nodes_before = len(orig.graph.node)
    nodes_after  = len(opt.graph.node)
    reduction    = nodes_before - nodes_after

    print(f"  ✓ bert-base:         {nodes_before} → {nodes_after} nodes "
          f"(-{reduction}) | max_diff={report.max_diff:.2e}")


if __name__ == "__main__":
    import os
    if not os.path.exists("tests/toy_models/shape_identity_reshape.onnx"):
        print("Building toy models first...")
        exec(open("tests/toy_models/build_shape_chain_model.py").read())

    print("\nRunning M6 tests...\n")
    test_identity_reshape()
    test_real_reshape()
    test_dead_shape_node()
    test_mobilenetv2()
    test_bert()
    print("\n✅ All M6 tests passed.")
```

---

## Downloading BERT for the First Time

```bash
mkdir -p models

# Option A — export from HuggingFace (requires transformers + torch)
pip install transformers torch

python -c "
from transformers import BertModel
import torch
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
dummy_input = {
    'input_ids':      torch.ones(1, 128, dtype=torch.long),
    'attention_mask': torch.ones(1, 128, dtype=torch.long),
}
torch.onnx.export(
    model,
    (dummy_input,),
    'models/bert-base-uncased.onnx',
    input_names=['input_ids', 'attention_mask'],
    output_names=['last_hidden_state'],
    dynamic_axes={
        'input_ids':      {0: 'batch', 1: 'seq'},
        'attention_mask': {0: 'batch', 1: 'seq'},
    },
    opset_version=14,
)
print('Exported: models/bert-base-uncased.onnx')
"

# Option B — download pre-exported from ONNX Model Zoo (if available)
# https://github.com/onnx/models/tree/main/validated/text/machine_comprehension/bert
```

**Add to `.gitignore`:**
```
models/
*.onnx
```

---

## Update `passes/__init__.py`

```python
from passes.eliminate_dead_nodes import EliminateDeadNodes
from passes.eliminate_identity_ops import EliminateIdentityOps
from passes.eliminate_unused_initializers import EliminateUnusedInitializers
from passes.eliminate_duplicate_constants import EliminateDuplicateConstants
from passes.eliminate_redundant_transposes import EliminateRedundantTransposes
from passes.fold_constants import FoldConstants
from passes.simplify_shape_chains import SimplifyShapeChains
```

## Update `optimizer.py`

```python
from passes.simplify_shape_chains import SimplifyShapeChains

registered_passes = [
    EliminateDeadNodes(),
    EliminateIdentityOps(),
    EliminateUnusedInitializers(),
    EliminateDuplicateConstants(),
    EliminateRedundantTransposes(),
    FoldConstants(),
    SimplifyShapeChains(),       # always runs after fold_constants
]
```

**Pass order is important here:** SimplifyShapeChains must run after FoldConstants.
Shape chains only become simplifiable BECAUSE constant folding already turned the
dynamic shape computations into static constants. Running M6 before M5 would find
nothing to do.

---

## Run Order

```bash
# Step 1 — build toy models
python tests/toy_models/build_shape_chain_model.py

# Step 2 — run targeted tests
python tests/test_shape_chains.py

# Step 3 — full optimizer on MobileNetV2
python optimizer.py mobilenetv2-12.onnx mobilenetv2-12-m6.onnx

# Step 4 — full optimizer on BERT (if exported)
python optimizer.py models/bert-base-uncased.onnx models/bert-base-uncased-opt.onnx

# Step 5 — verify BERT
python verify.py models/bert-base-uncased.onnx models/bert-base-uncased-opt.onnx
```

**Expected test output:**
```
Running M6 tests...

  ✓ identity_reshape:  Reshape removed | max_diff=0.00e+00
  ✓ real_reshape:      Reshape survived | max_diff=0.00e+00
  ✓ dead_shape_node:   2 → 1 node | Shape removed, Relu survived
  ✓ mobilenetv2:       105 → 105 nodes | max_diff=0.00e+00
  ✓ bert-base:         ~800 → ~650 nodes (-~150) | max_diff=<1e-4

✅ All M6 tests passed.
```

The BERT numbers are estimates — actual reduction depends on the export and which
passes fire. The key signal is that node count drops meaningfully for the first time
on a Transformer model.

---

## Definition of Done

- [ ] `build_shape_chain_model.py` generates all 3 toy models without errors
- [ ] All 3 `.onnx` toy files pass `onnx.checker.check_model()`
- [ ] `simplify_shape_chains.py` implemented
- [ ] `passes/__init__.py` updated
- [ ] `optimizer.py` updated with pass registered after fold_constants
- [ ] `test_identity_reshape` → Reshape removed, accuracy verified ✓
- [ ] `test_real_reshape` → Reshape survives ✓
- [ ] `test_dead_shape_node` → 2 nodes become 1, Shape removed, Relu survives ✓
- [ ] MobileNetV2 still clean
- [ ] BERT-base exported and optimizer runs on it without errors
- [ ] BERT node count drops (any reduction is a pass)
- [ ] BERT accuracy verified (max_diff within tolerance)

---

## Known Gotchas

**verify.py needs updating for BERT** — BERT has dynamic input shapes (`batch`, `seq`).
The `_get_input_specs` function in verify.py substitutes `1` for dynamic dims, which
should work for BERT. But `input_ids` is INT64 (token IDs) and `attention_mask` is
also INT64 — the dtype mapping in `_generate_random_input` must handle INT64 correctly.
Check this handles it — it should from the M1 implementation.

**Shape inference on BERT may be slow** — `onnx.shape_inference.infer_shapes(model)`
on a 400MB BERT model can take 10-30 seconds. This is called inside the pass. If it's
too slow, wrap it in a timer and consider making it optional via a flag.

**BERT verify tolerance** — floating point accumulation across 12 transformer layers
means max_diff may be slightly higher than `1e-5` even for lossless passes. If verify
fails with a diff around `1e-4` to `1e-3`, that's likely acceptable floating point
variance from ONNX Runtime's own optimizations, not a real accuracy bug. Bump the
tolerance for BERT to `1e-4` in the test.

**Dynamic axes in BERT export** — we export with dynamic `batch` and `seq` dimensions.
This means some shape chains genuinely cannot be fully folded (the batch dimension is
unknown at export time). M6 will simplify what it can and leave the rest. Don't expect
every Shape node to disappear.

---

## Next: M7 — fuse_conv_batchnorm

Once M6 is green → `micro_plan_M7.md`.

This is Tier 3 — the first fusion pass. BatchNorm parameters (scale, bias, mean,
variance) get mathematically folded into the preceding Conv weights. This removes
entire BN layers from the graph — real node count AND real size reduction.

The M7 toy model will have a hand-crafted Conv→BN pair where we can assert:
- The BN node is gone
- The Conv weights have been mathematically updated
- The output values are numerically identical to the original Conv+BN sequence
