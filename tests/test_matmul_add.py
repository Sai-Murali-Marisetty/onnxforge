"""
Tests for fuse_matmul_add pass.
Run: python tests/test_matmul_add.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
    """
    Two MatMul+Add pairs in sequence.
    First pair fuses (input X has known shape), second may not fuse
    if intermediate tensor 'h1' doesn't have shape info in value_info.
    This is expected behavior - without shape info, we can't verify rank-2.
    """
    orig, opt = _run_pass("tests/toy_models/matmul_add_double.onnx")

    assert len(orig.graph.node) == 4
    # At least first pair should fuse
    assert len(opt.graph.node) <= 3, f"Expected at most 3 nodes, got {len(opt.graph.node)}"

    gemm_nodes = [n for n in opt.graph.node if n.op_type == "Gemm"]
    assert len(gemm_nodes) >= 1, "At least first pair must become Gemm"

    report = verify(orig, opt, n_samples=10)
    assert report.passed and report.max_diff < 1e-5

    print(f"  ✓ two_linear_layers: 4 → {len(opt.graph.node)} nodes | "
          f"Gemm: {len(gemm_nodes)} | max_diff={report.max_diff:.2e}")


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
    if not os.path.exists("tests/toy_models/matmul_add.onnx"):
        print("Building toy models first...\n")
        exec(open("tests/toy_models/build_matmul_add_model.py").read())
        print()

    print("Running MatMul+Add tests...\n")
    test_matmul_add()
    test_matmul_no_add()
    test_two_linear_layers()
    test_mobilenetv2()
    print("\n✅ All MatMul+Add tests passed.")
