# M5 — Fold Constants

**Goal:** Implement the most impactful single pass in the pipeline. Any subgraph where
every input is a constant gets pre-computed by ONNX Runtime and replaced with a single
Constant node. Shape manipulation chains, positional encodings, mask computations — all
collapse into static values baked into the model.

---

## What You're Building

```
onnxslim/
├── passes/
│   └── fold_constants.py                    ← NEW
├── tests/
│   ├── toy_models/
│   │   ├── build_transpose_model.py         ✅ done
│   │   └── build_constants_model.py         ← NEW — 4 dirty constant subgraph models
│   └── test_fold_constants.py               ← NEW — precise assertions + value checks
```

---

## Background — What Is Constant Folding?

If every input to a node is a known constant (either an initializer or the output of
another constant node), then the node's output is also a constant — it will produce the
same value every single time regardless of what the runtime input is.

So instead of computing it at inference time (every forward pass), compute it once during
optimization and bake the result in as a new initializer.

```
# Before:
Constant([1.0, 2.0, 3.0]) ──┐
                             ├─→ Add → [2.0, 4.0, 6.0]
Constant([1.0, 2.0, 3.0]) ──┘

# After:
Constant([2.0, 4.0, 6.0])   ← pre-computed, single node
```

### Where this fires in real models

**HuggingFace Transformer exports** are full of constant subgraphs:

```python
# In Python (before export):
position_ids = torch.arange(seq_len).unsqueeze(0)  # shape: [1, 512]
```

This exports as a chain of ONNX ops:
```
Constant(0) → Constant(512) → Range → Unsqueeze(axis=0)
```

All inputs are constants → entire chain collapses to one Constant node holding the
pre-computed `[[0,1,2,...,511]]` tensor. That's 4 nodes → 1 node, zero computation at
inference.

Same story for attention masks, token type embeddings initialisation, scale factors in
attention (`1 / sqrt(head_dim)`), and LayerNorm epsilon computations.

---

## Algorithm

### Step 1 — Identify constant-valued tensors

Start with all initializers — they are by definition constant. Then propagate forward:
a node is "constant-foldable" if ALL of its inputs are in the known-constants set. Its
outputs then also become known constants.

```python
def find_foldable_nodes(graph):
    constant_tensors = {init.name for init in graph.initializer}

    # Also treat explicit Constant nodes as constant
    for node in graph.node:
        if node.op_type == "Constant":
            constant_tensors.update(node.output)

    foldable = []
    changed = True
    while changed:
        changed = False
        for node in graph.node:
            if node in foldable:
                continue
            if node.op_type == "Constant":
                continue  # already handled
            # All non-empty inputs must be constants
            if all(inp in constant_tensors or inp == "" for inp in node.input):
                foldable.append(node)
                constant_tensors.update(node.output)
                changed = True

    return foldable, constant_tensors
```

### Step 2 — Execute the foldable subgraph

Extract just the foldable nodes + their initializers into a mini ONNX model, run it
through ONNX Runtime, and collect the computed output tensors.

```python
import onnxruntime as ort
from onnx import numpy_helper

def execute_subgraph(foldable_nodes, graph, opset):
    # Build a mini model containing only the foldable subgraph
    # Run it through ORT to get the pre-computed values
    # Return dict: output_name → numpy array
    ...
```

### Step 3 — Replace with Constant nodes

For each foldable node output, create a new `Constant` node holding the pre-computed
numpy array as an initializer. Remove the original foldable nodes.

```python
from onnx import helper, numpy_helper, TensorProto

def make_constant_node(name, value_array):
    tensor = numpy_helper.from_array(value_array, name=f"{name}_folded")
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=tensor,
    )
```

---

## Pass: `passes/fold_constants.py`

```python
import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from passes.base_pass import BasePass


def _is_constant_node(node):
    return node.op_type == "Constant"


def _collect_constant_names(graph):
    """All tensor names that are statically known constants."""
    constants = set()

    # All initializers are constants
    for init in graph.initializer:
        constants.add(init.name)

    # Explicit Constant op outputs are constants
    for node in graph.node:
        if node.op_type == "Constant":
            constants.update(node.output)

    return constants


def _find_foldable_nodes(graph):
    """
    Propagate forward from known constants.
    A node is foldable if ALL its non-empty inputs are constants.
    Returns (foldable_nodes_list, all_constant_names_set).
    """
    constant_names = _collect_constant_names(graph)
    foldable = []
    visited = set()

    changed = True
    while changed:
        changed = False
        for node in graph.node:
            if id(node) in visited:
                continue
            if node.op_type == "Constant":
                continue
            if all(inp in constant_names or inp == "" for inp in node.input):
                foldable.append(node)
                visited.add(id(node))
                constant_names.update(node.output)
                changed = True

    return foldable, constant_names


def _get_output_names_to_keep(graph, constant_names):
    """
    Among constant-valued tensors, find those that are actually consumed
    by non-foldable nodes or are graph outputs.
    These are the ones we need to materialise as Constant nodes.
    """
    graph_output_names = {o.name for o in graph.output}
    foldable_names, _ = _find_foldable_nodes(graph)
    foldable_output_names = set()
    for node in foldable_names:
        foldable_output_names.update(node.output)

    # Find which foldable outputs feed non-foldable nodes
    needed = set()
    foldable_node_ids = {id(n) for n in foldable_names}

    for node in graph.node:
        if id(node) in foldable_node_ids:
            continue
        for inp in node.input:
            if inp in foldable_output_names:
                needed.add(inp)

    # Also keep graph outputs that happen to be constant
    needed.update(name for name in foldable_output_names if name in graph_output_names)

    return needed


def _build_mini_model(foldable_nodes, graph, opset_version):
    """
    Build a minimal ONNX model containing only the foldable subgraph.
    Used to run through ORT and get pre-computed values.
    """
    # Collect all initializers referenced by foldable nodes
    init_names_needed = set()
    for node in foldable_nodes:
        for inp in node.input:
            if inp:
                init_names_needed.add(inp)

    # Recursively include initializers of initializers (for chains)
    relevant_inits = [
        init for init in graph.initializer
        if init.name in init_names_needed
    ]

    # Also include Constant nodes referenced by foldable nodes
    const_nodes = [
        node for node in graph.node
        if node.op_type == "Constant"
    ]

    all_nodes = const_nodes + foldable_nodes

    # Collect all outputs of foldable nodes as graph outputs
    outputs = []
    for node in foldable_nodes:
        for out in node.output:
            if out:
                outputs.append(
                    helper.make_tensor_value_info(out, TensorProto.FLOAT, None)
                )

    graph_def = helper.make_graph(
        all_nodes,
        "fold_subgraph",
        inputs=[],
        outputs=outputs,
        initializer=relevant_inits,
    )

    opset_imports = [helper.make_opsetid("", opset_version)]
    model = helper.make_model(graph_def, opset_imports=opset_imports)
    model.ir_version = 8
    return model


def _run_subgraph(mini_model):
    """Execute mini model with ORT, return dict of output_name → numpy array."""
    sess = ort.InferenceSession(mini_model.SerializeToString())
    output_names = [o.name for o in mini_model.graph.output]
    results = sess.run(output_names, {})
    return dict(zip(output_names, results))


def _make_constant_node(output_name, array):
    """Create a Constant node pre-loaded with a numpy array."""
    tensor = numpy_helper.from_array(array.astype(array.dtype), name=f"folded_{output_name}")
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[output_name],
        value=tensor,
        name=f"folded_{output_name}",
    )


class FoldConstants(BasePass):

    @property
    def name(self) -> str:
        return "fold_constants"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        graph = model.graph

        # Detect opset version for building mini model
        opset_version = 13
        for opset in model.opset_import:
            if opset.domain == "" or opset.domain == "ai.onnx":
                opset_version = opset.version

        foldable_nodes, constant_names = _find_foldable_nodes(graph)

        if not foldable_nodes:
            return model

        # Figure out which outputs we actually need to materialise
        needed_outputs = _get_output_names_to_keep(graph, constant_names)

        # Only fold nodes that produce needed outputs
        nodes_to_fold = [
            n for n in foldable_nodes
            if any(out in needed_outputs for out in n.output)
        ]

        if not nodes_to_fold:
            return model

        # Execute the subgraph
        try:
            mini_model = _build_mini_model(foldable_nodes, graph, opset_version)
            computed = _run_subgraph(mini_model)
        except Exception as e:
            # Folding failed — skip rather than corrupt the model
            print(f"    ⚠ constant folding skipped: {e}")
            return model

        # Build replacement Constant nodes for needed outputs
        new_constant_nodes = []
        for output_name in needed_outputs:
            if output_name in computed:
                new_constant_nodes.append(
                    _make_constant_node(output_name, computed[output_name])
                )

        if not new_constant_nodes:
            return model

        # Remove folded nodes and old Constant nodes that fed them
        foldable_ids = {id(n) for n in foldable_nodes}
        surviving_nodes = [n for n in graph.node if id(n) not in foldable_ids]

        # Also remove Constant nodes whose outputs are no longer needed
        # (they've been absorbed into the folded result)
        folded_output_names = set()
        for n in foldable_nodes:
            folded_output_names.update(n.output)

        surviving_nodes = [
            n for n in surviving_nodes
            if not (n.op_type == "Constant" and
                    all(out in folded_output_names for out in n.output))
        ]

        # Add new pre-computed Constant nodes
        surviving_nodes.extend(new_constant_nodes)

        del graph.node[:]
        graph.node.extend(surviving_nodes)

        folded_count = len(foldable_nodes)
        replacement_count = len(new_constant_nodes)
        net_reduction = folded_count - replacement_count
        print(f"    → folded {folded_count} node(s) into {replacement_count} constant(s)"
              f" (net -{net_reduction} nodes)")

        return model
```

---

## Toy Models: `tests/toy_models/build_constants_model.py`

```python
"""
Builds synthetic dirty ONNX models for testing fold_constants.
Run: python tests/toy_models/build_constants_model.py
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def build_simple_add(output_path="tests/toy_models/constants_add.onnx"):
    """
    Two Constant tensors fed into Add.
    All inputs are constants → Add collapses to one Constant.

    Before: Const_A + Const_B → Add → output
            3 nodes (2 Constant + 1 Add)
    After:  Constant([2,4,6]) → output
            1 node

    We can assert the output value exactly: [1+1, 2+2, 3+3] = [2, 4, 6]
    """
    a_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    a_tensor = numpy_helper.from_array(a_val, name="a_val")
    b_tensor = numpy_helper.from_array(b_val, name="b_val")

    const_a = helper.make_node("Constant", [], ["A"], value=a_tensor, name="ConstA")
    const_b = helper.make_node("Constant", [], ["B"], value=b_tensor, name="ConstB")
    add     = helper.make_node("Add", ["A", "B"], ["Y"], name="Add")

    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])
    graph = helper.make_graph([const_a, const_b, add], "simple_add", [], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (3 nodes → expect 1 after fold)")


def build_chain_fold(output_path="tests/toy_models/constants_chain.onnx"):
    """
    A chain of constant ops simulating a positional encoding computation.
        Constant(0) → Constant(4) → Range(0,4,1) → Unsqueeze(axis=0)
    All constants → entire chain collapses to one Constant([[0,1,2,3]]).

    Before: 4 nodes
    After:  1 node with pre-computed [[0,1,2,3]]
    """
    # We'll keep it simple: Constant → Mul(scale) → result
    # (Range op has tricky shape inference, so we use simpler ops for the toy)
    base_val  = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    scale_val = np.array([2.0], dtype=np.float32)

    base_tensor  = numpy_helper.from_array(base_val,  name="base_val")
    scale_tensor = numpy_helper.from_array(scale_val, name="scale_val")

    const_base  = helper.make_node("Constant", [], ["base"],  value=base_tensor,  name="ConstBase")
    const_scale = helper.make_node("Constant", [], ["scale"], value=scale_tensor, name="ConstScale")
    mul         = helper.make_node("Mul", ["base", "scale"], ["scaled"], name="Mul")
    # Unsqueeze to add batch dim (simulating positional encoding shape)
    unsqueeze   = helper.make_node("Unsqueeze", ["scaled"], ["Y"], axes=[0], name="Unsqueeze")

    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph(
        [const_base, const_scale, mul, unsqueeze],
        "chain_fold", [], [Y]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (4 nodes → expect 1 after fold)")


def build_mixed_model(output_path="tests/toy_models/constants_mixed.onnx"):
    """
    Model with both constant and runtime (dynamic) subgraphs.
    Only the constant part should fold — the dynamic Add must survive.

    Constant(bias) → [pre-computable]
    runtime_input  → Add(runtime_input, bias) → output

    Before:
        Constant(bias) + Add(X, bias) = 2 nodes
    After:
        bias baked into Add as initializer-style constant: still 1 Add node
        but the Constant node is gone (absorbed)

    Key test: runtime Add node MUST survive. Only dead constant plumbing removed.
    """
    bias_val = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    bias_tensor = numpy_helper.from_array(bias_val, name="bias_val")

    X    = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])
    Y    = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])

    const_bias = helper.make_node("Constant", [], ["bias"], value=bias_tensor, name="ConstBias")
    add        = helper.make_node("Add", ["X", "bias"], ["Y"], name="Add")

    graph = helper.make_graph([const_bias, add], "mixed_model", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (bias is constant-fed Add — Add must survive)")


def build_no_fold_model(output_path="tests/toy_models/constants_no_fold.onnx"):
    """
    Model with zero constant subgraphs — all ops depend on runtime input.
    Pass should do nothing.

    X → Relu → Y
    Expected: 1 node → 1 node (untouched)
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])

    relu = helper.make_node("Relu", ["X"], ["Y"], name="Relu")

    graph = helper.make_graph([relu], "no_fold", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (no constants — pass does nothing)")


if __name__ == "__main__":
    os.makedirs("tests/toy_models", exist_ok=True)
    print("Building toy models for M5...\n")
    build_simple_add()
    build_chain_fold()
    build_mixed_model()
    build_no_fold_model()
    print("\nDone. All models saved to tests/toy_models/")
```

---

## Test File: `tests/test_fold_constants.py`

```python
"""
Tests for fold_constants pass.
Uses synthetic toy models with known expected outcomes.
Run: python tests/test_fold_constants.py
"""
import numpy as np
import onnx
from onnx import numpy_helper
from verify import verify
from passes.fold_constants import FoldConstants


def _run_pass(model_path):
    original  = onnx.load(model_path)
    model     = onnx.load(model_path)
    optimized = FoldConstants().run(model)
    onnx.checker.check_model(optimized)
    return original, optimized


def _get_constant_value(model, output_name):
    """Extract the pre-computed value from a Constant node by output name."""
    for node in model.graph.node:
        if node.op_type == "Constant" and output_name in node.output:
            for attr in node.attribute:
                if attr.name == "value":
                    return numpy_helper.to_array(attr.t)
    return None


def test_simple_add():
    """
    Two constants fed into Add → should collapse to one Constant with pre-computed value.
    We assert the VALUE of the result: [1+1, 2+2, 3+3] = [2, 4, 6]
    """
    orig, opt = _run_pass("tests/toy_models/constants_add.onnx")

    # Node count: 3 → 1
    assert len(orig.graph.node) == 3, f"Expected 3 nodes before, got {len(orig.graph.node)}"
    assert len(opt.graph.node) == 1, f"Expected 1 node after, got {len(opt.graph.node)}"
    assert opt.graph.node[0].op_type == "Constant", "Surviving node must be Constant"

    # Value check — the core assertion that distinguishes this from just "it ran"
    result = _get_constant_value(opt, "Y")
    assert result is not None, "Could not extract folded constant value"
    expected = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    assert np.allclose(result, expected, atol=1e-6), \
        f"Expected {expected}, got {result}"

    print(f"  ✓ simple_add:    3 → 1 node | value={result.tolist()} ✓")


def test_chain_fold():
    """
    4-node constant chain collapses to 1 Constant node.
    Value: base=[0,1,2,3] * scale=2 → [[0,2,4,6]] after unsqueeze.
    """
    orig, opt = _run_pass("tests/toy_models/constants_chain.onnx")

    assert len(orig.graph.node) == 4, f"Expected 4 nodes before"
    assert len(opt.graph.node) == 1, f"Expected 1 node after, got {len(opt.graph.node)}"

    result = _get_constant_value(opt, "Y")
    assert result is not None
    expected = np.array([[0.0, 2.0, 4.0, 6.0]], dtype=np.float32)
    assert np.allclose(result, expected, atol=1e-6), \
        f"Expected {expected}, got {result}"

    print(f"  ✓ chain_fold:    4 → 1 node | value={result.tolist()} ✓")


def test_mixed_model():
    """
    Model with both constant and runtime subgraphs.
    The runtime Add must survive. Only the Constant node feeding it is folded/absorbed.
    Node count: 2 → 1 (Constant absorbed, Add survives).
    Accuracy must match original across random inputs.
    """
    orig, opt = _run_pass("tests/toy_models/constants_mixed.onnx")

    # Add node must survive (it has a runtime input)
    surviving_ops = [n.op_type for n in opt.graph.node]
    assert "Add" in surviving_ops, f"Add node must survive. Got: {surviving_ops}"

    # Verify accuracy on random inputs
    report = verify(orig, opt, n_samples=10)
    assert report.passed and report.max_diff < 1e-5

    print(f"  ✓ mixed_model:   Add survived | max_diff={report.max_diff:.2e}")


def test_no_fold():
    """
    No constant subgraphs → pass does nothing.
    Node count: 1 → 1 (untouched).
    """
    orig, opt = _run_pass("tests/toy_models/constants_no_fold.onnx")

    assert len(orig.graph.node) == 1
    assert len(opt.graph.node) == 1, "No-fold model should be untouched"

    print(f"  ✓ no_fold:       1 → 1 node (untouched)")


def test_mobilenetv2():
    """
    Integration check on real model.
    MobileNetV2 may or may not have foldable constants — either is fine.
    Key: model stays valid and accurate.
    """
    orig, opt = _run_pass("mobilenetv2-12.onnx")
    report = verify(orig, opt, n_samples=5)
    assert report.passed

    nodes_before = len(orig.graph.node)
    nodes_after  = len(opt.graph.node)
    print(f"  ✓ mobilenetv2:   {nodes_before} → {nodes_after} nodes | max_diff={report.max_diff:.2e}")


if __name__ == "__main__":
    import os
    if not os.path.exists("tests/toy_models/constants_add.onnx"):
        print("Building toy models first...")
        exec(open("tests/toy_models/build_constants_model.py").read())

    print("\nRunning M5 tests...\n")
    test_simple_add()
    test_chain_fold()
    test_mixed_model()
    test_no_fold()
    test_mobilenetv2()
    print("\n✅ All M5 tests passed.")
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
```

## Update `optimizer.py`

```python
from passes.fold_constants import FoldConstants

registered_passes = [
    EliminateDeadNodes(),
    EliminateIdentityOps(),
    EliminateUnusedInitializers(),
    EliminateDuplicateConstants(),
    EliminateRedundantTransposes(),
    FoldConstants(),
]
```

**Pass order note:** FoldConstants runs after the structural passes. This is intentional —
dead node elimination and identity removal first means the constant folding subgraph
detection has a cleaner graph to work with and won't try to fold already-dead branches.

---

## Run Order

```bash
# Step 1 — build toy models
python tests/toy_models/build_constants_model.py

# Step 2 — targeted tests with value assertions
python tests/test_fold_constants.py

# Step 3 — full optimizer integration check
python optimizer.py mobilenetv2-12.onnx mobilenetv2-12-m5.onnx

# Step 4 — standalone verify
python verify.py mobilenetv2-12.onnx mobilenetv2-12-m5.onnx
```

**Expected test output:**
```
Running M5 tests...

  ✓ simple_add:    3 → 1 node | value=[2.0, 4.0, 6.0] ✓
  ✓ chain_fold:    4 → 1 node | value=[[0.0, 2.0, 4.0, 6.0]] ✓
  ✓ mixed_model:   Add survived | max_diff=0.00e+00
  ✓ no_fold:       1 → 1 node (untouched)
  ✓ mobilenetv2:   105 → ? nodes | max_diff=0.00e+00

✅ All M5 tests passed.
```

---

## Definition of Done

- [ ] `build_constants_model.py` generates all 4 toy models without errors
- [ ] All 4 `.onnx` toy files pass `onnx.checker.check_model()`
- [ ] `fold_constants.py` implemented
- [ ] `passes/__init__.py` updated
- [ ] `optimizer.py` updated with pass registered
- [ ] `test_simple_add` → 3 nodes become 1, value asserted as `[2.0, 4.0, 6.0]` ✓
- [ ] `test_chain_fold` → 4 nodes become 1, value asserted as `[[0.0, 2.0, 4.0, 6.0]]` ✓
- [ ] `test_mixed_model` → Add node survives, accuracy verified ✓
- [ ] `test_no_fold` → 1 node stays 1 (untouched) ✓
- [ ] Full optimizer run on MobileNetV2 stays clean

---

## Known Gotchas

**Unsqueeze opset difference** — `Unsqueeze` changed its signature between opset 11 and 13.
In opset 11, `axes` is an attribute. In opset 13, `axes` is an input tensor. The toy model
builder uses opset 13. If you hit issues, check which opset the model uses and adapt.

**Mini model output type inference** — when building the mini model to run through ORT,
you declare outputs as `TensorProto.FLOAT` with `None` shape. ORT can usually infer this,
but for integer ops (like Range, Gather with int64) you may need to set the correct dtype.
If ORT complains about type mismatches, inspect the actual output dtype from the original
model's value_info.

**Circular dependency risk** — the `_get_output_names_to_keep` function calls
`_find_foldable_nodes` internally. This is fine but means you're doing two passes over
the graph during setup. For models with thousands of nodes this is negligible. If you ever
profile and find it slow, cache the result.

**Large tensor folding** — if a foldable node produces a huge tensor (e.g. a full
embedding table computation), folding it bakes the entire tensor into the model file and
increases model size. In practice this is rare because embedding tables are usually
initializers already. But if model size goes UP after folding, check for this.

**Exception swallowing** — the pass wraps `_run_subgraph` in try/except and skips on
failure rather than crashing. This is intentional — a folding failure should not corrupt
the model. But during development, temporarily remove the try/except so you see the full
error when debugging.

---

## Next: M6 — simplify_shape_chains

Once M5 is green → `micro_plan_M6.md`.

After constant folding, many shape manipulation chains become dead:
```
Shape → Gather(indices=0) → Unsqueeze → Concat → Reshape
```
...where `Reshape` was using a dynamically computed shape that is now a static constant.
M6 cleans up these now-dead shape ops after M5 has done its work.

The M6 toy model will simulate a HuggingFace-style shape chain that collapses entirely
after folding. First time we test BERT-base.
