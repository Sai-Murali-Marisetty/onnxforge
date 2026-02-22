# M7 — Fuse Conv + BatchNorm

**Goal:** Implement the first fusion pass. BatchNorm parameters (scale, bias, mean,
variance) are mathematically folded into the preceding Conv weights and bias. The BN
layer disappears entirely. First time you'll see real node count AND model size reduction
on a production model (EfficientNet-B0).

---

## What You're Building

```
onnxslim/
├── passes/
│   └── fuse_conv_batchnorm.py               ← NEW
├── tests/
│   ├── toy_models/
│   │   └── build_conv_bn_model.py           ← NEW — 3 dirty Conv+BN models
│   └── test_conv_batchnorm.py               ← NEW — value assertions on fused weights
models/
└── efficientnet-b0.onnx                     ← DOWNLOAD — first model with real BN layers
```

---

## Background — The Math

BatchNorm normalises its input using running statistics computed during training:

```
BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
```

When BN follows a Conv, `x` is the Conv output. Since Conv is linear, we can absorb
the entire BN computation into the Conv weights and bias — turning two ops into one.

### Derivation

```
Conv output:   y = W * x + b          (where * is convolution, b is bias)
BN output:     z = gamma * (y - mean) / sqrt(var + eps) + beta

Substituting:  z = gamma * (W * x + b - mean) / sqrt(var + eps) + beta

Let:  scale = gamma / sqrt(var + eps)

Then: z = scale * W * x + scale * (b - mean) + beta
        = (scale * W) * x + (scale * (b - mean) + beta)

So:   new_weight = W * scale          (per output channel)
      new_bias   = (b - mean) * scale + beta
```

This is provably lossless — the math is exact. The only source of numerical difference
is floating point precision, which will be within 1e-5.

### Per-channel scaling

Conv weights have shape `[out_channels, in_channels, kH, kW]`.
BN parameters (`gamma`, `beta`, `mean`, `var`) each have shape `[out_channels]`.
The scale factor is applied per output channel — so you reshape scale to
`[out_channels, 1, 1, 1]` before multiplying with weights.

```python
scale      = gamma / np.sqrt(var + eps)          # [out_channels]
scale_4d   = scale.reshape(-1, 1, 1, 1)          # [out_channels, 1, 1, 1]
new_weight = weight * scale_4d                    # [out_channels, in_channels, kH, kW]
new_bias   = (bias - mean) * scale + beta        # [out_channels]
```

---

## Pass: `passes/fuse_conv_batchnorm.py`

```python
import onnx
import numpy as np
from onnx import numpy_helper
from passes.base_pass import BasePass


def _get_initializer(graph, name):
    """Fetch a named initializer as a numpy array. Returns None if not found."""
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def _set_initializer(graph, name, array):
    """Update or create a named initializer from a numpy array."""
    tensor = numpy_helper.from_array(array.astype(np.float32), name=name)
    for i, init in enumerate(graph.initializer):
        if init.name == name:
            graph.initializer[i].CopyFrom(tensor)
            return
    graph.initializer.append(tensor)


def _find_bn_consumer(graph, conv_output):
    """
    Find a BatchNormalization node that directly consumes the given tensor name.
    Returns the BN node or None.
    """
    for node in graph.node:
        if node.op_type == "BatchNormalization":
            if node.input[0] == conv_output:
                return node
    return None


def _fuse_conv_bn(conv_node, bn_node, graph):
    """
    Fold BN into Conv. Returns True if fusion succeeded, False otherwise.
    Modifies graph in-place.
    """
    # Conv inputs: [X, W] or [X, W, B]
    weight_name = conv_node.input[1]
    bias_name   = conv_node.input[2] if len(conv_node.input) > 2 else None

    # BN inputs: [X, scale(gamma), bias(beta), mean, var]
    gamma_name = bn_node.input[1]
    beta_name  = bn_node.input[2]
    mean_name  = bn_node.input[3]
    var_name   = bn_node.input[4]

    # Load all arrays
    weight = _get_initializer(graph, weight_name)
    gamma  = _get_initializer(graph, gamma_name)
    beta   = _get_initializer(graph, beta_name)
    mean   = _get_initializer(graph, mean_name)
    var    = _get_initializer(graph, var_name)

    if any(x is None for x in [weight, gamma, beta, mean, var]):
        return False  # dynamic BN params — cannot fuse statically

    bias = _get_initializer(graph, bias_name) if bias_name else np.zeros(weight.shape[0], dtype=np.float32)

    # Extract epsilon from BN attributes
    eps = 1e-5
    for attr in bn_node.attribute:
        if attr.name == "epsilon":
            eps = attr.f
            break

    # Compute fused weights
    scale      = gamma / np.sqrt(var + eps)         # [out_channels]
    scale_4d   = scale.reshape(-1, 1, 1, 1)         # broadcast over spatial dims
    new_weight = weight * scale_4d                   # [out_channels, in_channels, kH, kW]
    new_bias   = (bias - mean) * scale + beta        # [out_channels]

    # Update Conv weight initializer in-place
    _set_initializer(graph, weight_name, new_weight)

    # Set Conv bias
    fused_bias_name = bias_name if bias_name else f"{weight_name}_fused_bias"
    _set_initializer(graph, fused_bias_name, new_bias)

    # Update Conv node inputs to include bias if it didn't before
    if not bias_name:
        conv_node.input.append(fused_bias_name)

    # Rewire Conv output → BN output (skip BN entirely)
    bn_output = bn_node.output[0]
    conv_node.output[0] = bn_output

    return True


class FuseConvBatchnorm(BasePass):

    @property
    def name(self) -> str:
        return "fuse_conv_batchnorm"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        graph = model.graph
        graph_output_names = {o.name for o in graph.output}

        bn_nodes_to_remove = []
        fused_count = 0

        for node in graph.node:
            if node.op_type != "Conv":
                continue

            conv_output = node.output[0]

            # Skip if Conv output is a graph output (BN output replaces it — safe)
            # Actually this IS safe — we rewire Conv output to BN output name
            bn_node = _find_bn_consumer(graph, conv_output)
            if bn_node is None:
                continue

            # BN output must not be split across multiple consumers of the BN node
            # (in practice BN always has exactly one output used)
            success = _fuse_conv_bn(node, bn_node, graph)
            if success:
                bn_nodes_to_remove.append(id(bn_node))
                fused_count += 1

        if bn_nodes_to_remove:
            bn_id_set = set(bn_nodes_to_remove)
            new_nodes = [n for n in graph.node if id(n) not in bn_id_set]
            del graph.node[:]
            graph.node.extend(new_nodes)
            print(f"    → fused {fused_count} Conv+BN pair(s)")

        return model
```

---

## Toy Models: `tests/toy_models/build_conv_bn_model.py`

```python
"""
Builds synthetic dirty ONNX models for testing fuse_conv_batchnorm.
Run: python tests/toy_models/build_conv_bn_model.py
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def _add_initializer(graph, name, array):
    tensor = numpy_helper.from_array(array.astype(np.float32), name=name)
    graph.initializer.append(tensor)


def build_conv_bn_pair(output_path="tests/toy_models/conv_bn_pair.onnx"):
    """
    Single Conv → BatchNorm pair.
    Conv: 1 input channel, 2 output channels, 3x3 kernel
    BN:   2 channels

    Before: 2 nodes (Conv + BN)
    After:  1 node  (Conv with fused weights)

    We set known weight values so we can assert the fused result exactly.
    """
    # Conv weights: [out=2, in=1, kH=3, kW=3] — all ones for easy math
    W = np.ones((2, 1, 3, 3), dtype=np.float32)
    # Conv bias: zeros
    b = np.zeros(2, dtype=np.float32)

    # BN params: gamma=2, beta=1, mean=0, var=3 (so scale = 2/sqrt(3+1e-5) ≈ 1.0)
    # Using simple values: gamma=1, beta=0, mean=0, var=1 → scale=1/(sqrt(1+eps))≈1
    # This means fused weights ≈ W * 1 = W (easy to verify)
    gamma = np.array([1.0, 1.0], dtype=np.float32)
    beta  = np.array([0.0, 0.0], dtype=np.float32)
    mean  = np.array([0.0, 0.0], dtype=np.float32)
    var   = np.array([1.0, 1.0], dtype=np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 3, 3])

    conv = helper.make_node(
        "Conv", ["X", "W", "b"], ["conv_out"],
        kernel_shape=[3, 3], name="Conv"
    )
    bn = helper.make_node(
        "BatchNormalization",
        ["conv_out", "gamma", "beta", "mean", "var"],
        ["Y"],
        epsilon=1e-5, name="BN"
    )

    graph = helper.make_graph([conv, bn], "conv_bn_pair", [X], [Y])
    _add_initializer(graph, "W",     W)
    _add_initializer(graph, "b",     b)
    _add_initializer(graph, "gamma", gamma)
    _add_initializer(graph, "beta",  beta)
    _add_initializer(graph, "mean",  mean)
    _add_initializer(graph, "var",   var)

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (Conv+BN → 2 nodes, expect 1 after fuse)")


def build_conv_no_bn(output_path="tests/toy_models/conv_no_bn.onnx"):
    """
    Conv with no following BN — pass should do nothing.
    Before: 1 node (Conv)
    After:  1 node (untouched)
    """
    W = np.ones((2, 1, 3, 3), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 3, 3])

    conv = helper.make_node("Conv", ["X", "W", "b"], ["Y"],
                            kernel_shape=[3, 3], name="Conv")

    graph = helper.make_graph([conv], "conv_no_bn", [X], [Y])
    _add_initializer(graph, "W", W)
    _add_initializer(graph, "b", b)

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (Conv only — pass does nothing)")


def build_two_conv_bn_pairs(output_path="tests/toy_models/conv_bn_double.onnx"):
    """
    Two sequential Conv+BN pairs.
    Before: 4 nodes (Conv BN Conv BN)
    After:  2 nodes (Conv Conv — both BNs fused)
    """
    W1 = np.ones((4, 1, 3, 3), dtype=np.float32)
    b1 = np.zeros(4, dtype=np.float32)
    W2 = np.ones((4, 4, 1, 1), dtype=np.float32)
    b2 = np.zeros(4, dtype=np.float32)

    # BN params for both (identity-ish: gamma=1, beta=0, mean=0, var=1)
    gamma1 = np.ones(4,  dtype=np.float32)
    beta1  = np.zeros(4, dtype=np.float32)
    mean1  = np.zeros(4, dtype=np.float32)
    var1   = np.ones(4,  dtype=np.float32)

    gamma2 = np.ones(4,  dtype=np.float32) * 2   # non-trivial scale
    beta2  = np.ones(4,  dtype=np.float32)        # non-zero bias
    mean2  = np.ones(4,  dtype=np.float32) * 0.5
    var2   = np.ones(4,  dtype=np.float32)

    X = helper.make_tensor_value_info("X",  TensorProto.FLOAT, [1, 1, 5, 5])
    Y = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, [1, 4, 3, 3])

    conv1 = helper.make_node("Conv", ["X",       "W1", "b1"], ["c1"], kernel_shape=[3, 3], name="Conv1")
    bn1   = helper.make_node("BatchNormalization", ["c1", "g1", "bt1", "m1", "v1"], ["bn1_out"], epsilon=1e-5, name="BN1")
    conv2 = helper.make_node("Conv", ["bn1_out", "W2", "b2"], ["c2"], kernel_shape=[1, 1], name="Conv2")
    bn2   = helper.make_node("BatchNormalization", ["c2", "g2", "bt2", "m2", "v2"], ["Y"],      epsilon=1e-5, name="BN2")

    graph = helper.make_graph([conv1, bn1, conv2, bn2], "double_conv_bn", [X], [Y])
    for name, arr in [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2),
                      ("g1", gamma1), ("bt1", beta1), ("m1", mean1), ("v1", var1),
                      ("g2", gamma2), ("bt2", beta2), ("m2", mean2), ("v2", var2)]:
        _add_initializer(graph, name, arr)

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (4 nodes → expect 2 after fuse)")


if __name__ == "__main__":
    os.makedirs("tests/toy_models", exist_ok=True)
    print("Building toy models for M7...\n")
    build_conv_bn_pair()
    build_conv_no_bn()
    build_two_conv_bn_pairs()
    print("\nDone. All models saved to tests/toy_models/")
```

---

## Test File: `tests/test_conv_batchnorm.py`

```python
"""
Tests for fuse_conv_batchnorm pass.
Uses toy models with known weight values — we assert fused weights are numerically correct.
Run: python tests/test_conv_batchnorm.py
"""
import numpy as np
import onnx
from onnx import numpy_helper
from verify import verify
from passes.fuse_conv_batchnorm import FuseConvBatchnorm


def _run_pass(model_path):
    original  = onnx.load(model_path)
    model     = onnx.load(model_path)
    optimized = FuseConvBatchnorm().run(model)
    onnx.checker.check_model(optimized)
    return original, optimized


def _get_initializer(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def test_conv_bn_pair():
    """
    Single Conv+BN → 2 nodes become 1.
    Fused weights should be mathematically correct.
    Output values must match original within tolerance.
    """
    orig, opt = _run_pass("tests/toy_models/conv_bn_pair.onnx")

    # Node count: 2 → 1
    assert len(orig.graph.node) == 2
    assert len(opt.graph.node) == 1, f"Expected 1 node, got {len(opt.graph.node)}"
    assert opt.graph.node[0].op_type == "Conv", "Surviving node must be Conv"

    # Verify no BN nodes remain
    bn_nodes = [n for n in opt.graph.node if n.op_type == "BatchNormalization"]
    assert len(bn_nodes) == 0, "BN nodes must all be removed"

    # Weight value check:
    # gamma=1, beta=0, mean=0, var=1, eps=1e-5
    # scale = 1 / sqrt(1 + 1e-5) ≈ 0.999995
    # new_weight = W * scale ≈ W (since W=all ones)
    # new_bias = (0 - 0) * scale + 0 = 0
    fused_weight = _get_initializer(opt, "W")
    assert fused_weight is not None
    expected_scale = 1.0 / np.sqrt(1.0 + 1e-5)
    expected_weight = np.ones((2, 1, 3, 3), dtype=np.float32) * expected_scale
    assert np.allclose(fused_weight, expected_weight, atol=1e-5), \
        f"Fused weights don't match expected.\nGot:      {fused_weight.flatten()[:4]}\nExpected: {expected_weight.flatten()[:4]}"

    # Accuracy: run random inputs through both
    report = verify(orig, opt, n_samples=10)
    assert report.passed and report.max_diff < 1e-4

    print(f"  ✓ conv_bn_pair:      2 → 1 node | weights verified | max_diff={report.max_diff:.2e}")


def test_conv_no_bn():
    """Conv without BN — pass does nothing."""
    orig, opt = _run_pass("tests/toy_models/conv_no_bn.onnx")

    assert len(opt.graph.node) == 1, "Should be untouched"
    assert opt.graph.node[0].op_type == "Conv"

    report = verify(orig, opt, n_samples=5)
    assert report.passed

    print(f"  ✓ conv_no_bn:        1 → 1 node (untouched) | max_diff={report.max_diff:.2e}")


def test_two_conv_bn_pairs():
    """
    Two sequential Conv+BN pairs → 4 nodes become 2.
    Both BN nodes removed. Both Conv nodes have fused weights.
    """
    orig, opt = _run_pass("tests/toy_models/conv_bn_double.onnx")

    assert len(orig.graph.node) == 4
    assert len(opt.graph.node) == 2, f"Expected 2 nodes, got {len(opt.graph.node)}"

    bn_nodes = [n for n in opt.graph.node if n.op_type == "BatchNormalization"]
    assert len(bn_nodes) == 0, "All BN nodes must be removed"

    conv_nodes = [n for n in opt.graph.node if n.op_type == "Conv"]
    assert len(conv_nodes) == 2, "Both Conv nodes must survive"

    report = verify(orig, opt, n_samples=10)
    assert report.passed and report.max_diff < 1e-4

    print(f"  ✓ two_conv_bn_pairs: 4 → 2 nodes | both BNs fused | max_diff={report.max_diff:.2e}")


def test_mobilenetv2():
    """
    MobileNetV2 has Conv+BN patterns — should see real reduction.
    """
    orig, opt = _run_pass("mobilenetv2-12.onnx")
    report = verify(orig, opt, n_samples=5)
    assert report.passed

    nodes_before = len(orig.graph.node)
    nodes_after  = len(opt.graph.node)
    bn_before    = sum(1 for n in orig.graph.node if n.op_type == "BatchNormalization")
    bn_after     = sum(1 for n in opt.graph.node  if n.op_type == "BatchNormalization")

    print(f"  ✓ mobilenetv2:       {nodes_before} → {nodes_after} nodes | "
          f"BN: {bn_before} → {bn_after} | max_diff={report.max_diff:.2e}")


def test_efficientnet():
    """
    EfficientNet-B0 — primary target for this pass.
    Dense Conv+BN+Swish structure. Should see significant BN removal.

    Download:
    python -c \"
    import torchvision, torch
    model = torchvision.models.efficientnet_b0(pretrained=True).eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy, 'models/efficientnet-b0.onnx',
                      opset_version=13,
                      input_names=['input'], output_names=['output'])
    \"
    """
    import os
    eff_path = "models/efficientnet-b0.onnx"
    if not os.path.exists(eff_path):
        print(f"  ⚠ EfficientNet model not found at {eff_path} — skipping")
        print(f"    Run the export command in the docstring above to generate it")
        return

    orig, opt = _run_pass(eff_path)
    report = verify(orig, opt, n_samples=3)
    assert report.passed

    nodes_before = len(orig.graph.node)
    nodes_after  = len(opt.graph.node)
    bn_before    = sum(1 for n in orig.graph.node if n.op_type == "BatchNormalization")
    bn_after     = sum(1 for n in opt.graph.node  if n.op_type == "BatchNormalization")
    size_before  = orig.ByteSize() / 1024 / 1024
    size_after   = opt.ByteSize()  / 1024 / 1024

    print(f"  ✓ efficientnet-b0:   {nodes_before} → {nodes_after} nodes | "
          f"BN: {bn_before} → {bn_after} | "
          f"size: {size_before:.1f}MB → {size_after:.1f}MB | "
          f"max_diff={report.max_diff:.2e}")


if __name__ == "__main__":
    import os
    if not os.path.exists("tests/toy_models/conv_bn_pair.onnx"):
        print("Building toy models first...")
        exec(open("tests/toy_models/build_conv_bn_model.py").read())

    print("\nRunning M7 tests...\n")
    test_conv_bn_pair()
    test_conv_no_bn()
    test_two_conv_bn_pairs()
    test_mobilenetv2()
    test_efficientnet()
    print("\n✅ All M7 tests passed.")
```

---

## Downloading EfficientNet-B0

```bash
mkdir -p models

pip install torchvision torch

python -c "
import torchvision, torch
model = torchvision.models.efficientnet_b0(weights='DEFAULT').eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, 'models/efficientnet-b0.onnx',
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
)
print('Exported: models/efficientnet-b0.onnx')
"
```

---

## Update `passes/__init__.py`

```python
from passes.fuse_conv_batchnorm import FuseConvBatchnorm
```

## Update `optimizer.py`

```python
from passes.fuse_conv_batchnorm import FuseConvBatchnorm

registered_passes = [
    EliminateDeadNodes(),
    EliminateIdentityOps(),
    EliminateUnusedInitializers(),
    EliminateDuplicateConstants(),
    EliminateRedundantTransposes(),
    FoldConstants(),
    SimplifyShapeChains(),
    FuseConvBatchnorm(),            # Tier 3 — first fusion pass
]
```

---

## Run Order

```bash
# Step 1 — build toy models
python tests/toy_models/build_conv_bn_model.py

# Step 2 — targeted tests with weight value assertions
python tests/test_conv_batchnorm.py

# Step 3 — full pipeline on MobileNetV2
python optimizer.py mobilenetv2-12.onnx mobilenetv2-12-m7.onnx

# Step 4 — full pipeline on EfficientNet (if exported)
python optimizer.py models/efficientnet-b0.onnx models/efficientnet-b0-opt.onnx
python verify.py models/efficientnet-b0.onnx models/efficientnet-b0-opt.onnx
```

**Expected test output:**
```
Running M7 tests...

  ✓ conv_bn_pair:      2 → 1 node | weights verified | max_diff=~1e-7
  ✓ conv_no_bn:        1 → 1 node (untouched) | max_diff=0.00e+00
  ✓ two_conv_bn_pairs: 4 → 2 nodes | both BNs fused | max_diff=~1e-7
  ✓ mobilenetv2:       105 → ~70 nodes | BN: 35 → 0 | max_diff=~1e-6
  ✓ efficientnet-b0:   ~350 → ~230 nodes | BN: ~80 → 0 | size: ~21MB → ~20MB

✅ All M7 tests passed.
```

MobileNetV2 has 35 BN layers — all should fuse. EfficientNet has more, with deeper
BN chains. Both models should show real node count AND size reduction for the first time.

---

## Definition of Done

- [ ] `build_conv_bn_model.py` generates all 3 toy models without errors
- [ ] All 3 `.onnx` toy files pass `onnx.checker.check_model()`
- [ ] `fuse_conv_batchnorm.py` implemented
- [ ] `passes/__init__.py` updated
- [ ] `optimizer.py` updated with pass registered
- [ ] `test_conv_bn_pair` → 2 nodes become 1, fused weights verified numerically ✓
- [ ] `test_conv_no_bn` → 1 node stays 1 (untouched) ✓
- [ ] `test_two_conv_bn_pairs` → 4 nodes become 2, both BNs removed ✓
- [ ] MobileNetV2: BN count drops to 0, accuracy verified ✓
- [ ] EfficientNet exported and optimizer runs on it (BN count drops, size drops)

---

## Known Gotchas

**Conv with no bias** — many Convs in exported models have no bias term (only `[X, W]`
as inputs, not `[X, W, B]`). The pass handles this by creating a zero bias. Make sure
you append the new bias name to `conv_node.input` and add it as an initializer —
otherwise the Conv node will reference a name that doesn't exist.

**Depthwise Conv** — in MobileNetV2 and EfficientNet, depthwise separable convolutions
have weight shape `[out_channels, 1, kH, kW]` (groups == out_channels). The BN fusion
math is identical — scale broadcasting still works because `[out_channels, 1, 1, 1]`
broadcasts correctly against `[out_channels, 1, kH, kW]`.

**BN with multiple outputs** — `BatchNormalization` in training mode has 5 outputs
(Y, mean, var, saved_mean, saved_var). In inference mode (which is what you're
optimising), only output[0] (Y) is used. The pass only rewires `bn_node.output[0]`.
If a model was exported in training mode with multiple BN outputs consumed, the pass
will leave it alone (the check `_find_bn_consumer` finds the node, but the other
outputs would become dangling). Add a guard: only fuse if `len(used_bn_outputs) == 1`.

**Accuracy tolerance** — BN fusion introduces floating point rounding errors.
`1e-4` is the right tolerance for this pass, not `1e-5`. The error comes from the
difference between `float32(a * b + c)` computed inline vs precomputed. Perfectly
normal and expected.
