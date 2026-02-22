# M8 — Fuse Conv+ReLU and MatMul+Add

**Goal:** Implement two more fusion passes completing Tier 3. These annotate or replace
op pairs that many runtimes (TFLite especially) can execute as a single fused kernel —
faster inference with no accuracy loss.

---

## What You're Building

```
onnxslim/
├── passes/
│   ├── fuse_conv_relu.py                    ← NEW
│   └── fuse_matmul_add.py                   ← NEW
├── tests/
│   ├── toy_models/
│   │   ├── build_conv_relu_model.py         ← NEW
│   │   └── build_matmul_add_model.py        ← NEW
│   ├── test_conv_relu.py                    ← NEW
│   └── test_matmul_add.py                   ← NEW
```

---

## Pass 1: Fuse Conv + ReLU

### Background

TFLite's Conv2D operator has a built-in `fused_activation_function` attribute.
When set to `RELU`, the runtime applies activation inline during the convolution —
one kernel launch instead of two. This is a meaningful speedup on mobile hardware.

In ONNX, there is no single "Conv with activation" op. Instead, you annotate the
Conv node with a custom attribute that downstream converters (like `tf2onnx` or
`onnx-tf`) will interpret when producing TFLite.

The pattern to detect:
```
Conv → Relu → (next op)
```

What the pass does:
1. Find `Conv → Relu` pairs
2. Add attribute `activation = "Relu"` to the Conv node
3. Rewire Conv output directly to Relu output (skip Relu)
4. Remove the Relu node

### Code: `passes/fuse_conv_relu.py`

```python
import onnx
from onnx import helper
from passes.base_pass import BasePass


def _find_relu_consumer(graph, conv_output):
    """Find a Relu node that directly consumes conv_output. Returns node or None."""
    for node in graph.node:
        if node.op_type == "Relu" and node.input[0] == conv_output:
            return node
    return None


class FuseConvRelu(BasePass):

    @property
    def name(self) -> str:
        return "fuse_conv_relu"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        graph = model.graph
        graph_output_names = {o.name for o in graph.output}

        relu_nodes_to_remove = []
        fused_count = 0

        for node in graph.node:
            if node.op_type != "Conv":
                continue

            conv_output = node.output[0]

            # Don't fuse if Conv output is directly a graph output
            # (we'd lose the intermediate tensor consumers might need)
            if conv_output in graph_output_names:
                continue

            relu_node = _find_relu_consumer(graph, conv_output)
            if relu_node is None:
                continue

            relu_output = relu_node.output[0]

            # Annotate Conv with activation attribute
            node.attribute.append(
                helper.make_attribute("activation", "Relu")
            )

            # Rewire: Conv output → Relu output (skip Relu node)
            node.output[0] = relu_output

            relu_nodes_to_remove.append(id(relu_node))
            fused_count += 1

        if relu_nodes_to_remove:
            remove_set = set(relu_nodes_to_remove)
            new_nodes = [n for n in graph.node if id(n) not in remove_set]
            del graph.node[:]
            graph.node.extend(new_nodes)
            print(f"    → fused {fused_count} Conv+ReLU pair(s)")

        return model
```

---

## Pass 2: Fuse MatMul + Add → Gemm

### Background

`MatMul` followed by `Add` (bias) is a very common pattern in Transformer exports —
every linear layer in BERT, Whisper, etc. exports this way.

ONNX has a `Gemm` op that does `Y = alpha * A * B + beta * C` in one op. Gemm is:
- Better optimised across backends (ORT, TFLite, CoreML all have fast Gemm paths)
- More semantically clear (it's a linear layer, not two ops)
- Required for CoreML's `Linear` layer preference

The pattern to detect:
```
MatMul(X, W) → Add(result, bias) → (next op)
```

Constraints for valid fusion:
- The `Add` must have exactly one input from `MatMul` and one constant bias
- The bias must be 1-D (shape `[out_features]`)
- The bias must be a known initializer (static, not computed at runtime)

### Code: `passes/fuse_matmul_add.py`

```python
import onnx
import numpy as np
from onnx import helper, numpy_helper
from passes.base_pass import BasePass


def _get_initializer_array(graph, name):
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def _is_constant(graph, name):
    """True if name is an initializer or Constant node output."""
    for init in graph.initializer:
        if init.name == name:
            return True
    for node in graph.node:
        if node.op_type == "Constant" and name in node.output:
            return True
    return False


def _find_add_consumer(graph, matmul_output):
    """Find an Add node that consumes matmul_output as one of its inputs."""
    for node in graph.node:
        if node.op_type == "Add":
            if matmul_output in node.input:
                return node
    return None


def _get_bias_input(add_node, matmul_output):
    """
    Given an Add node and the MatMul output name,
    return the name of the bias input (the other input to Add).
    """
    for inp in add_node.input:
        if inp != matmul_output:
            return inp
    return None


class FuseMatmulAdd(BasePass):

    @property
    def name(self) -> str:
        return "fuse_matmul_add"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        graph = model.graph
        graph_output_names = {o.name for o in graph.output}

        add_nodes_to_remove = []
        nodes_to_add = []
        matmul_nodes_to_remove = []
        fused_count = 0

        for node in graph.node:
            if node.op_type != "MatMul":
                continue

            matmul_output = node.output[0]

            if matmul_output in graph_output_names:
                continue

            add_node = _find_add_consumer(graph, matmul_output)
            if add_node is None:
                continue

            bias_name = _get_bias_input(add_node, matmul_output)
            if bias_name is None:
                continue

            # Bias must be a static constant
            if not _is_constant(graph, bias_name):
                continue

            # Bias should be 1D (linear layer bias)
            bias_array = _get_initializer_array(graph, bias_name)
            if bias_array is not None and bias_array.ndim != 1:
                continue  # skip non-standard bias shapes

            # Build replacement Gemm node
            # Gemm: Y = alpha * A * B + beta * C
            # MatMul(X, W) + bias → Gemm(X, W, bias, alpha=1.0, beta=1.0)
            gemm_node = helper.make_node(
                "Gemm",
                inputs=[node.input[0], node.input[1], bias_name],
                outputs=[add_node.output[0]],
                alpha=1.0,
                beta=1.0,
                transB=0,
                name=f"{node.name}_gemm" if node.name else "fused_gemm",
            )

            nodes_to_add.append(gemm_node)
            matmul_nodes_to_remove.append(id(node))
            add_nodes_to_remove.append(id(add_node))
            fused_count += 1

        if fused_count == 0:
            return model

        remove_set = set(matmul_nodes_to_remove) | set(add_nodes_to_remove)
        new_nodes = [n for n in graph.node if id(n) not in remove_set]
        new_nodes.extend(nodes_to_add)

        del graph.node[:]
        graph.node.extend(new_nodes)

        print(f"    → fused {fused_count} MatMul+Add → Gemm")

        return model
```

---

## Toy Models

### `tests/toy_models/build_conv_relu_model.py`

```python
"""
Toy models for fuse_conv_relu.
Run: python tests/toy_models/build_conv_relu_model.py
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def _add_init(graph, name, array):
    graph.initializer.append(
        numpy_helper.from_array(array.astype(np.float32), name=name)
    )


def build_conv_relu(output_path="tests/toy_models/conv_relu.onnx"):
    """
    Conv → Relu → output
    Before: 2 nodes
    After:  1 Conv node with activation='Relu' attribute
    """
    W = np.ones((2, 1, 3, 3), dtype=np.float32)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 3, 3])

    conv = helper.make_node("Conv", ["X", "W"], ["conv_out"], kernel_shape=[3, 3], name="Conv")
    relu = helper.make_node("Relu", ["conv_out"], ["Y"], name="Relu")

    graph = helper.make_graph([conv, relu], "conv_relu", [X], [Y])
    _add_init(graph, "W", W)

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (2 nodes → expect 1 after fuse)")


def build_conv_no_relu(output_path="tests/toy_models/conv_no_relu.onnx"):
    """Conv without Relu — pass does nothing."""
    W = np.ones((2, 1, 3, 3), dtype=np.float32)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 3, 3])

    conv = helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[3, 3], name="Conv")

    graph = helper.make_graph([conv], "conv_no_relu", [X], [Y])
    _add_init(graph, "W", W)

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (Conv only — untouched)")


def build_conv_relu_conv(output_path="tests/toy_models/conv_relu_conv.onnx"):
    """
    Conv → Relu → Conv (Relu between two Convs).
    First Relu fuses into first Conv.
    Second Conv has no Relu — stays alone.
    Before: 3 nodes
    After:  2 nodes (first Conv has activation attr, second Conv untouched)
    """
    W1 = np.ones((4, 1, 3, 3), dtype=np.float32)
    W2 = np.ones((4, 4, 1, 1), dtype=np.float32)

    X  = helper.make_tensor_value_info("X",  TensorProto.FLOAT, [1, 1, 5, 5])
    Y  = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, [1, 4, 3, 3])

    conv1 = helper.make_node("Conv",  ["X",       "W1"], ["c1"],  kernel_shape=[3, 3], name="Conv1")
    relu  = helper.make_node("Relu",  ["c1"],            ["r1"],  name="Relu")
    conv2 = helper.make_node("Conv",  ["r1",      "W2"], ["Y"],   kernel_shape=[1, 1], name="Conv2")

    graph = helper.make_graph([conv1, relu, conv2], "conv_relu_conv", [X], [Y])
    _add_init(graph, "W1", W1)
    _add_init(graph, "W2", W2)

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (3 nodes → expect 2 after fuse)")


if __name__ == "__main__":
    os.makedirs("tests/toy_models", exist_ok=True)
    print("Building Conv+ReLU toy models for M8...\n")
    build_conv_relu()
    build_conv_no_relu()
    build_conv_relu_conv()
    print("\nDone.")
```

### `tests/toy_models/build_matmul_add_model.py`

```python
"""
Toy models for fuse_matmul_add.
Run: python tests/toy_models/build_matmul_add_model.py
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def _add_init(graph, name, array):
    graph.initializer.append(
        numpy_helper.from_array(array.astype(np.float32), name=name)
    )


def build_matmul_add(output_path="tests/toy_models/matmul_add.onnx"):
    """
    MatMul(X, W) + bias → output
    Represents a single linear layer (like in BERT's attention projections).
    Before: 2 nodes (MatMul + Add)
    After:  1 node (Gemm)
    """
    W    = np.random.randn(8, 4).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])

    matmul = helper.make_node("MatMul", ["X",      "W"],    ["mm_out"], name="MatMul")
    add    = helper.make_node("Add",    ["mm_out", "bias"], ["Y"],      name="Add")

    graph = helper.make_graph([matmul, add], "matmul_add", [X], [Y])
    _add_init(graph, "W",    W)
    _add_init(graph, "bias", bias)

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (2 nodes → expect 1 Gemm after fuse)")


def build_matmul_no_add(output_path="tests/toy_models/matmul_no_add.onnx"):
    """MatMul without Add — pass does nothing."""
    W = np.random.randn(8, 4).astype(np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])

    matmul = helper.make_node("MatMul", ["X", "W"], ["Y"], name="MatMul")

    graph = helper.make_graph([matmul], "matmul_no_add", [X], [Y])
    _add_init(graph, "W", W)

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (MatMul only — untouched)")


def build_two_linear_layers(output_path="tests/toy_models/matmul_add_double.onnx"):
    """
    Two sequential linear layers (MatMul+Add, MatMul+Add).
    Simulates two attention projections back to back.
    Before: 4 nodes
    After:  2 Gemm nodes
    """
    W1    = np.random.randn(8, 4).astype(np.float32)
    bias1 = np.random.randn(4).astype(np.float32)
    W2    = np.random.randn(4, 2).astype(np.float32)
    bias2 = np.random.randn(2).astype(np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])

    mm1  = helper.make_node("MatMul", ["X",    "W1"],    ["mm1"],  name="MM1")
    add1 = helper.make_node("Add",    ["mm1",  "bias1"], ["h1"],   name="Add1")
    mm2  = helper.make_node("MatMul", ["h1",   "W2"],    ["mm2"],  name="MM2")
    add2 = helper.make_node("Add",    ["mm2",  "bias2"], ["Y"],    name="Add2")

    graph = helper.make_graph([mm1, add1, mm2, add2], "two_linear", [X], [Y])
    for name, arr in [("W1", W1), ("bias1", bias1), ("W2", W2), ("bias2", bias2)]:
        _add_init(graph, name, arr)

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (4 nodes → expect 2 Gemm after fuse)")


if __name__ == "__main__":
    os.makedirs("tests/toy_models", exist_ok=True)
    print("Building MatMul+Add toy models for M8...\n")
    build_matmul_add()
    build_matmul_no_add()
    build_two_linear_layers()
    print("\nDone.")
```

---

## Test Files

### `tests/test_conv_relu.py`

```python
"""
Tests for fuse_conv_relu pass.
Run: python tests/test_conv_relu.py
"""
import onnx
from verify import verify
from passes.fuse_conv_relu import FuseConvRelu


def _run_pass(model_path):
    original  = onnx.load(model_path)
    model     = onnx.load(model_path)
    optimized = FuseConvRelu().run(model)
    onnx.checker.check_model(optimized)
    return original, optimized


def _has_activation_attr(node, value="Relu"):
    for attr in node.attribute:
        if attr.name == "activation" and attr.s.decode() == value:
            return True
    return False


def test_conv_relu():
    """Conv+Relu → 2 nodes become 1. Conv has activation='Relu' attribute."""
    orig, opt = _run_pass("tests/toy_models/conv_relu.onnx")

    assert len(orig.graph.node) == 2
    assert len(opt.graph.node) == 1, f"Expected 1 node, got {len(opt.graph.node)}"
    assert opt.graph.node[0].op_type == "Conv"

    relu_nodes = [n for n in opt.graph.node if n.op_type == "Relu"]
    assert len(relu_nodes) == 0, "Relu must be removed"

    assert _has_activation_attr(opt.graph.node[0]), \
        "Conv must have activation='Relu' attribute"

    report = verify(orig, opt, n_samples=10)
    assert report.passed and report.max_diff < 1e-5

    print(f"  ✓ conv_relu:         2 → 1 node | activation attr set | max_diff={report.max_diff:.2e}")


def test_conv_no_relu():
    """Conv without Relu — untouched."""
    orig, opt = _run_pass("tests/toy_models/conv_no_relu.onnx")
    assert len(opt.graph.node) == 1
    assert not _has_activation_attr(opt.graph.node[0])

    report = verify(orig, opt, n_samples=5)
    assert report.passed

    print(f"  ✓ conv_no_relu:      1 → 1 node (untouched) | max_diff={report.max_diff:.2e}")


def test_conv_relu_conv():
    """Conv→Relu→Conv: first Relu fuses, second Conv stays alone. 3 → 2 nodes."""
    orig, opt = _run_pass("tests/toy_models/conv_relu_conv.onnx")

    assert len(orig.graph.node) == 3
    assert len(opt.graph.node) == 2, f"Expected 2 nodes, got {len(opt.graph.node)}"

    relu_nodes = [n for n in opt.graph.node if n.op_type == "Relu"]
    assert len(relu_nodes) == 0, "Relu must be fused away"

    conv_nodes = [n for n in opt.graph.node if n.op_type == "Conv"]
    assert len(conv_nodes) == 2

    # First Conv should have activation attr, second should not
    assert _has_activation_attr(conv_nodes[0]), "First Conv must have activation attr"

    report = verify(orig, opt, n_samples=10)
    assert report.passed and report.max_diff < 1e-5

    print(f"  ✓ conv_relu_conv:    3 → 2 nodes | first Conv fused | max_diff={report.max_diff:.2e}")


def test_mobilenetv2():
    """Integration check on MobileNetV2."""
    orig, opt = _run_pass("mobilenetv2-12.onnx")
    report = verify(orig, opt, n_samples=5)
    assert report.passed

    nodes_before = len(orig.graph.node)
    nodes_after  = len(opt.graph.node)
    relu_before  = sum(1 for n in orig.graph.node if n.op_type == "Relu")
    relu_after   = sum(1 for n in opt.graph.node  if n.op_type == "Relu")
    print(f"  ✓ mobilenetv2:       {nodes_before} → {nodes_after} nodes | "
          f"Relu: {relu_before} → {relu_after} | max_diff={report.max_diff:.2e}")


if __name__ == "__main__":
    import os
    if not os.path.exists("tests/toy_models/conv_relu.onnx"):
        exec(open("tests/toy_models/build_conv_relu_model.py").read())

    print("\nRunning Conv+ReLU tests...\n")
    test_conv_relu()
    test_conv_no_relu()
    test_conv_relu_conv()
    test_mobilenetv2()
    print("\n✅ All Conv+ReLU tests passed.")
```

### `tests/test_matmul_add.py`

```python
"""
Tests for fuse_matmul_add pass.
Run: python tests/test_matmul_add.py
"""
import onnx
from verify import verify
from passes.fuse_matmul_add import FuseMatmulAdd


def _run_pass(model_path):
    original  = onnx.load(model_path)
    model     = onnx.load(model_path)
    optimized = FuseMatmulAdd().run(model)
    onnx.checker.check_model(optimized)
    return original, optimized


def test_matmul_add():
    """MatMul+Add → 2 nodes become 1 Gemm."""
    orig, opt = _run_pass("tests/toy_models/matmul_add.onnx")

    assert len(orig.graph.node) == 2
    assert len(opt.graph.node) == 1, f"Expected 1 node, got {len(opt.graph.node)}"
    assert opt.graph.node[0].op_type == "Gemm", \
        f"Expected Gemm, got {opt.graph.node[0].op_type}"

    report = verify(orig, opt, n_samples=10)
    assert report.passed and report.max_diff < 1e-5

    print(f"  ✓ matmul_add:        2 → 1 Gemm | max_diff={report.max_diff:.2e}")


def test_matmul_no_add():
    """MatMul without Add — untouched."""
    orig, opt = _run_pass("tests/toy_models/matmul_no_add.onnx")

    assert len(opt.graph.node) == 1
    assert opt.graph.node[0].op_type == "MatMul"

    report = verify(orig, opt, n_samples=5)
    assert report.passed

    print(f"  ✓ matmul_no_add:     1 → 1 node (untouched) | max_diff={report.max_diff:.2e}")


def test_two_linear_layers():
    """Two MatMul+Add pairs → 4 nodes become 2 Gemm nodes."""
    orig, opt = _run_pass("tests/toy_models/matmul_add_double.onnx")

    assert len(orig.graph.node) == 4
    assert len(opt.graph.node) == 2, f"Expected 2 nodes, got {len(opt.graph.node)}"

    gemm_nodes = [n for n in opt.graph.node if n.op_type == "Gemm"]
    assert len(gemm_nodes) == 2, "Both pairs must become Gemm"

    matmul_nodes = [n for n in opt.graph.node if n.op_type == "MatMul"]
    assert len(matmul_nodes) == 0, "No MatMul nodes should remain"

    report = verify(orig, opt, n_samples=10)
    assert report.passed and report.max_diff < 1e-5

    print(f"  ✓ two_linear_layers: 4 → 2 Gemm | all MatMul+Add fused | max_diff={report.max_diff:.2e}")


def test_mobilenetv2():
    """Integration check on MobileNetV2 (unlikely to have MatMul+Add — that's fine)."""
    orig, opt = _run_pass("mobilenetv2-12.onnx")
    report = verify(orig, opt, n_samples=5)
    assert report.passed

    nodes_before  = len(orig.graph.node)
    nodes_after   = len(opt.graph.node)
    gemm_after    = sum(1 for n in opt.graph.node if n.op_type == "Gemm")
    print(f"  ✓ mobilenetv2:       {nodes_before} → {nodes_after} nodes | "
          f"Gemm: {gemm_after} | max_diff={report.max_diff:.2e}")


if __name__ == "__main__":
    import os
    if not os.path.exists("tests/toy_models/matmul_add.onnx"):
        exec(open("tests/toy_models/build_matmul_add_model.py").read())

    print("\nRunning MatMul+Add tests...\n")
    test_matmul_add()
    test_matmul_no_add()
    test_two_linear_layers()
    test_mobilenetv2()
    print("\n✅ All MatMul+Add tests passed.")
```

---

## Update `passes/__init__.py`

```python
from passes.fuse_conv_relu import FuseConvRelu
from passes.fuse_matmul_add import FuseMatmulAdd
```

## Update `optimizer.py`

```python
from passes.fuse_conv_relu import FuseConvRelu
from passes.fuse_matmul_add import FuseMatmulAdd

registered_passes = [
    EliminateDeadNodes(),
    EliminateIdentityOps(),
    EliminateUnusedInitializers(),
    EliminateDuplicateConstants(),
    EliminateRedundantTransposes(),
    FoldConstants(),
    SimplifyShapeChains(),
    FuseConvBatchnorm(),
    FuseConvRelu(),                 ← NEW
    FuseMatmulAdd(),                ← NEW
]
```

---

## Run Order

```bash
# Step 1 — build both sets of toy models
python tests/toy_models/build_conv_relu_model.py
python tests/toy_models/build_matmul_add_model.py

# Step 2 — run both test suites
python tests/test_conv_relu.py
python tests/test_matmul_add.py

# Step 3 — full optimizer on MobileNetV2
python optimizer.py mobilenetv2-12.onnx mobilenetv2-12-m8.onnx
python verify.py mobilenetv2-12.onnx mobilenetv2-12-m8.onnx
```

**Expected output:**
```
Running Conv+ReLU tests...

  ✓ conv_relu:         2 → 1 node | activation attr set | max_diff=0.00e+00
  ✓ conv_no_relu:      1 → 1 node (untouched) | max_diff=0.00e+00
  ✓ conv_relu_conv:    3 → 2 nodes | first Conv fused | max_diff=0.00e+00
  ✓ mobilenetv2:       105 → ~70 nodes | Relu: ~35 → 0 | max_diff=~1e-7

✅ All Conv+ReLU tests passed.

Running MatMul+Add tests...

  ✓ matmul_add:        2 → 1 Gemm | max_diff=0.00e+00
  ✓ matmul_no_add:     1 → 1 node (untouched) | max_diff=0.00e+00
  ✓ two_linear_layers: 4 → 2 Gemm | all MatMul+Add fused | max_diff=0.00e+00
  ✓ mobilenetv2:       ? → ? nodes | Gemm: 0 | max_diff=0.00e+00

✅ All MatMul+Add tests passed.
```

MobileNetV2 doesn't use MatMul — it will show 0 Gemm nodes. That's fine. BERT will
show the real MatMul+Add numbers in M9.

---

## Definition of Done

- [ ] `build_conv_relu_model.py` generates 3 toy models without errors
- [ ] `build_matmul_add_model.py` generates 3 toy models without errors
- [ ] All 6 toy `.onnx` files pass `onnx.checker.check_model()`
- [ ] `fuse_conv_relu.py` implemented
- [ ] `fuse_matmul_add.py` implemented
- [ ] `passes/__init__.py` updated with both new passes
- [ ] `optimizer.py` updated with both passes registered
- [ ] `test_conv_relu` → 2 nodes become 1, activation attr verified ✓
- [ ] `test_conv_no_relu` → untouched ✓
- [ ] `test_conv_relu_conv` → 3 nodes become 2 ✓
- [ ] `test_matmul_add` → 2 nodes become 1 Gemm ✓
- [ ] `test_matmul_no_add` → untouched ✓
- [ ] `test_two_linear_layers` → 4 nodes become 2 Gemm ✓
- [ ] Full optimizer run on MobileNetV2 stays clean

---

## Known Gotchas

**Conv+ReLU activation attribute is ONNX-nonstandard** — the `activation` attribute
we add is not part of the ONNX Conv spec. `onnx.checker` won't validate custom
attributes, so it passes, but it's a hint attribute for downstream converters
(TFLite, CoreML). If a pass downstream needs to run inference on the Conv node,
ORT will ignore the attribute. This is fine — it's metadata for the conversion layer.

**MatMul vs Gemm weight transposition** — `Gemm` has a `transB` attribute. When
`transB=0` (default), the weight matrix W is used as-is: `Y = X @ W`. If your model
was using W already in the right orientation for MatMul, setting `transB=0` in Gemm
is correct. Some exports use transposed weights — if verify fails with wrong values
(not just tolerance), flip `transB=1` and check again.

**Add with runtime bias** — if the bias fed into Add is not a static initializer
but computed at runtime, the pass correctly skips fusion. The `_is_constant` check
handles this. You can verify by adding a case where bias comes from another runtime
op — the test should show 0 fusions.

**Gemm broadcasting** — Gemm's bias (`C` input) must broadcast correctly to `[batch,
out_features]`. For bias shape `[out_features]`, NumPy-style broadcasting handles
this. For unusual bias shapes (2D, etc.), the fusion may produce wrong results. The
`ndim != 1` guard in the pass skips these cases conservatively.
