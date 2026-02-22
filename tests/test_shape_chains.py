"""
Tests for simplify_shape_chains pass.
Run: python tests/test_shape_chains.py
"""
import numpy as np
import onnx
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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


if __name__ == "__main__":
    if not os.path.exists("tests/toy_models/shape_identity_reshape.onnx"):
        print("Building toy models first...\n")
        exec(open("tests/toy_models/build_shape_chain_model.py").read())
        print()

    print("Running M6 tests...\n")
    test_identity_reshape()
    test_real_reshape()
    test_dead_shape_node()
    test_mobilenetv2()
    print("\n✅ All M6 tests passed.")
