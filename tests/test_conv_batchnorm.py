"""
Tests for fuse_conv_batchnorm pass.
Uses toy models with known weight values — we assert fused weights are numerically correct.
Run: python tests/test_conv_batchnorm.py
"""
import numpy as np
import onnx
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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


if __name__ == "__main__":
    if not os.path.exists("tests/toy_models/conv_bn_pair.onnx"):
        print("Building toy models first...\n")
        exec(open("tests/toy_models/build_conv_bn_model.py").read())
        print()

    print("Running M7 tests...\n")
    test_conv_bn_pair()
    test_conv_no_bn()
    test_two_conv_bn_pairs()
    test_mobilenetv2()
    print("\n✅ All M7 tests passed.")
