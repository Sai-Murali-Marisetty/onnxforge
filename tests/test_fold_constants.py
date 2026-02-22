"""
Tests for fold_constants pass.
Uses synthetic toy models with known expected outcomes.
Run: python tests/test_fold_constants.py
"""
import numpy as np
import onnx
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
    5-node constant chain collapses to 1 Constant node.
    Value: base=[0,1,2,3] * scale=2 → [[0,2,4,6]] after unsqueeze.
    """
    orig, opt = _run_pass("tests/toy_models/constants_chain.onnx")

    assert len(orig.graph.node) == 5, f"Expected 5 nodes before, got {len(orig.graph.node)}"
    assert len(opt.graph.node) == 1, f"Expected 1 node after, got {len(opt.graph.node)}"

    result = _get_constant_value(opt, "Y")
    assert result is not None
    expected = np.array([[0.0, 2.0, 4.0, 6.0]], dtype=np.float32)
    assert np.allclose(result, expected, atol=1e-6), \
        f"Expected {expected}, got {result}"

    print(f"  ✓ chain_fold:    5 → 1 node | value={result.tolist()} ✓")


def test_mixed_model():
    """
    Model with both constant and runtime subgraphs.
    The runtime Add must survive. Only the Constant node feeding it stays.
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
    if not os.path.exists("tests/toy_models/constants_add.onnx"):
        print("Building toy models first...\n")
        exec(open("tests/toy_models/build_constants_model.py").read())
        print()

    print("Running M5 tests...\n")
    test_simple_add()
    test_chain_fold()
    test_mixed_model()
    test_no_fold()
    test_mobilenetv2()
    print("\n✅ All M5 tests passed.")
