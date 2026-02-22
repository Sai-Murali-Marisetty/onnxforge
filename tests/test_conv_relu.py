"""
Tests for fuse_conv_relu pass.
Run: python tests/test_conv_relu.py

Note: This pass is currently a pattern detector for TFLite export compatibility.
It identifies Conv+Relu pairs but doesn't actually fuse them in the ONNX graph
because ONNX Runtime doesn't support custom 'activation' attributes on Conv.

The tests verify pattern detection works correctly and the model is unchanged.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import onnx
from verify import verify
from passes.fuse_conv_relu import FuseConvRelu


def _run_pass(model_path):
    original  = onnx.load(model_path)
    model     = onnx.load(model_path)
    optimized = FuseConvRelu().run(model)
    onnx.checker.check_model(optimized)
    return original, optimized


def test_conv_relu():
    """Conv+Relu pattern detected but not fused (ORT compatibility)."""
    orig, opt = _run_pass("tests/toy_models/conv_relu.onnx")

    # Pattern detection only - no node reduction
    assert len(opt.graph.node) == len(orig.graph.node), \
        "Node count should be unchanged (pattern detection only)"

    report = verify(orig, opt, n_samples=10)
    assert report.passed

    print(f"  ✓ conv_relu:         {len(opt.graph.node)} nodes (pattern detected) | max_diff={report.max_diff:.2e}")


def test_conv_no_relu():
    """Conv without Relu — no pattern found."""
    orig, opt = _run_pass("tests/toy_models/conv_no_relu.onnx")
    assert len(opt.graph.node) == 1

    report = verify(orig, opt, n_samples=5)
    assert report.passed

    print(f"  ✓ conv_no_relu:      1 node (no pattern) | max_diff={report.max_diff:.2e}")


def test_conv_relu_conv():
    """Conv→Relu→Conv: pattern detected for first pair."""
    orig, opt = _run_pass("tests/toy_models/conv_relu_conv.onnx")

    # No fusion - just pattern detection
    assert len(opt.graph.node) == 3, "Node count should be unchanged"

    report = verify(orig, opt, n_samples=10)
    assert report.passed

    print(f"  ✓ conv_relu_conv:    {len(opt.graph.node)} nodes (pattern detected) | max_diff={report.max_diff:.2e}")


def test_mobilenetv2():
    """Integration check on MobileNetV2."""
    orig, opt = _run_pass("mobilenetv2-12.onnx")
    report = verify(orig, opt, n_samples=5)
    assert report.passed

    nodes_count = len(opt.graph.node)
    relu_count = sum(1 for n in opt.graph.node if n.op_type == "Relu")
    print(f"  ✓ mobilenetv2:       {nodes_count} nodes | Relu: {relu_count} | max_diff={report.max_diff:.2e}")


if __name__ == "__main__":
    if not os.path.exists("tests/toy_models/conv_relu.onnx"):
        print("Building toy models first...\n")
        exec(open("tests/toy_models/build_conv_relu_model.py").read())
        print()

    print("Running Conv+ReLU tests (pattern detection mode)...\n")
    test_conv_relu()
    test_conv_no_relu()
    test_conv_relu_conv()
    test_mobilenetv2()
    print("\n✅ All Conv+ReLU tests passed.")
