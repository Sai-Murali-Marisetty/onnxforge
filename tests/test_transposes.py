"""
Tests for eliminate_redundant_transposes.
Uses synthetic toy models with known expected outcomes.
Run: python tests/test_transposes.py
"""
import onnx
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from verify import verify
from passes.eliminate_redundant_transposes import EliminateRedundantTransposes


def _run_pass(model_path):
    original  = onnx.load(model_path)
    model     = onnx.load(model_path)
    optimized = EliminateRedundantTransposes().run(model)
    report    = verify(original, optimized, n_samples=10)
    onnx.checker.check_model(optimized)
    return original, optimized, report


def test_cancelling_pair():
    orig, opt, report = _run_pass("tests/toy_models/transpose_cancelling.onnx")
    assert len(orig.graph.node) == 2, "Expected 2 nodes before"
    assert len(opt.graph.node) == 0, f"Expected 0 nodes after, got {len(opt.graph.node)}"
    assert report.passed and report.max_diff < 1e-5
    print(f"  ✓ cancelling_pair:  2 → 0 nodes | max_diff={report.max_diff:.2e}")


def test_mergeable_chain():
    orig, opt, report = _run_pass("tests/toy_models/transpose_mergeable.onnx")
    assert len(orig.graph.node) == 2
    assert len(opt.graph.node) == 1, f"Expected 1 node after, got {len(opt.graph.node)}"

    surviving = opt.graph.node[0]
    perm = list(surviving.attribute[0].ints)
    # p1=[0,2,1,3], p2=[0,1,3,2] -> composed=[0,2,3,1]
    assert perm == [0, 2, 3, 1], f"Expected composed perm [0,2,3,1], got {perm}"

    assert report.passed and report.max_diff < 1e-5
    print(f"  ✓ mergeable_chain:  2 → 1 node | perm={perm} | max_diff={report.max_diff:.2e}")


def test_clean_model():
    orig, opt, report = _run_pass("tests/toy_models/transpose_clean.onnx")
    assert len(orig.graph.node) == 1
    assert len(opt.graph.node) == 1, "Clean model should be untouched"
    assert report.passed
    print(f"  ✓ clean_model:      1 → 1 node (untouched) | max_diff={report.max_diff:.2e}")


def test_triple_chain():
    orig, opt, report = _run_pass("tests/toy_models/transpose_triple.onnx")
    assert len(orig.graph.node) == 3
    assert len(opt.graph.node) == 1, f"Expected 1 node after, got {len(opt.graph.node)}"
    assert report.passed and report.max_diff < 1e-5
    print(f"  ✓ triple_chain:     3 → 1 node | max_diff={report.max_diff:.2e}")


def test_mobilenetv2():
    """Integration check — pass runs cleanly on a real model."""
    orig, opt, report = _run_pass("mobilenetv2-12.onnx")
    assert report.passed
    nodes_before = len(orig.graph.node)
    nodes_after  = len(opt.graph.node)
    print(f"  ✓ mobilenetv2:      {nodes_before} → {nodes_after} nodes | max_diff={report.max_diff:.2e}")


if __name__ == "__main__":
    # Build toy models if they don't exist
    if not os.path.exists("tests/toy_models/transpose_cancelling.onnx"):
        print("Building toy models first...\n")
        exec(open("tests/toy_models/build_transpose_model.py").read())
        print()

    print("Running M4 tests...\n")
    test_cancelling_pair()
    test_mergeable_chain()
    test_clean_model()
    test_triple_chain()
    test_mobilenetv2()
    print("\n✅ All M4 tests passed.")
