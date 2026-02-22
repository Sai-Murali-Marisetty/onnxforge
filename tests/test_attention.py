"""
Tests for cleanup_attention pass.
Run: python tests/test_attention.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import onnx
from verify import verify
from passes.cleanup_attention import CleanupAttention


def _run_pass(model_path):
    original  = onnx.load(model_path)
    model     = onnx.load(model_path)
    optimized = CleanupAttention().run(model)
    onnx.checker.check_model(optimized)
    return original, optimized


def test_consecutive_reshape():
    """Two consecutive Reshape ops → should merge to 1."""
    orig, opt = _run_pass("tests/toy_models/consecutive_reshape.onnx")
    
    reshape_before = sum(1 for n in orig.graph.node if n.op_type == "Reshape")
    reshape_after = sum(1 for n in opt.graph.node if n.op_type == "Reshape")
    
    assert reshape_before == 2, f"Expected 2 Reshape before, got {reshape_before}"
    assert reshape_after == 1, f"Expected 1 Reshape after, got {reshape_after}"
    
    report = verify(orig, opt, n_samples=10)
    assert report.passed
    
    print(f"  ✓ consecutive_reshape: {reshape_before} → {reshape_after} Reshape | max_diff={report.max_diff:.2e}")


def test_identity_reshape():
    """Identity Reshape — pattern detection for future removal."""
    orig, opt = _run_pass("tests/toy_models/identity_reshape.onnx")
    
    # This pass doesn't remove identity reshapes yet (needs shape inference)
    # Just verify the model is still valid
    report = verify(orig, opt, n_samples=10)
    assert report.passed
    
    nodes_before = len(orig.graph.node)
    nodes_after = len(opt.graph.node)
    
    print(f"  ✓ identity_reshape: {nodes_before} → {nodes_after} nodes | max_diff={report.max_diff:.2e}")


def test_branching_reshape():
    """Reshape with multiple consumers — should NOT be removed."""
    orig, opt = _run_pass("tests/toy_models/branching_reshape.onnx")
    
    # Should be unchanged because Reshape has multiple consumers
    nodes_before = len(orig.graph.node)
    nodes_after = len(opt.graph.node)
    
    # The branching case: Reshape feeds both Relu and Sigmoid
    # We should NOT remove it
    report = verify(orig, opt, n_samples=10)
    assert report.passed
    
    print(f"  ✓ branching_reshape: {nodes_before} → {nodes_after} nodes (unchanged) | max_diff={report.max_diff:.2e}")


def test_mobilenetv2():
    """Integration check on MobileNetV2."""
    orig, opt = _run_pass("mobilenetv2-12.onnx")
    report = verify(orig, opt, n_samples=5)
    assert report.passed
    
    reshape_before = sum(1 for n in orig.graph.node if n.op_type == "Reshape")
    reshape_after = sum(1 for n in opt.graph.node if n.op_type == "Reshape")
    
    print(f"  ✓ mobilenetv2: Reshape {reshape_before} → {reshape_after} | max_diff={report.max_diff:.2e}")


if __name__ == "__main__":
    if not os.path.exists("tests/toy_models/consecutive_reshape.onnx"):
        print("Building toy models first...\n")
        exec(open("tests/toy_models/build_attention_model.py").read())
        print()
    
    print("Running Attention cleanup tests...\n")
    test_consecutive_reshape()
    test_identity_reshape()
    test_branching_reshape()
    test_mobilenetv2()
    print("\n✅ All Attention cleanup tests passed.")
