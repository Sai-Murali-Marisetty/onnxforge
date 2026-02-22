"""
MobileNetV2 Tests — M2 and M3
Verifies optimization passes:
  1. Reduce or maintain node count (never increase)
  2. Preserve accuracy within tolerance
  3. Produce a valid ONNX graph
"""
import onnx
import onnxruntime as ort
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optimizer import optimize
from verify import verify
from passes import (
    EliminateDeadNodes,
    EliminateIdentityOps,
    EliminateUnusedInitializers,
    EliminateDuplicateConstants,
)

MODEL_PATH = "mobilenetv2-12.onnx"
OUTPUT_PATH_M2 = "mobilenetv2-12-m2.onnx"
OUTPUT_PATH_M3 = "mobilenetv2-12-m3.onnx"

def test_m2_mobilenetv2():
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}. Download from ONNX Model Zoo."

    passes = [EliminateDeadNodes(), EliminateIdentityOps()]

    report = optimize(
        model_path=MODEL_PATH,
        output_path=OUTPUT_PATH_M2,
        passes=passes,
        verify_each_pass=True,
        n_verify_samples=10,
    )

    # Node count must not increase
    assert report["nodes_after"] <= report["nodes_before"], \
        f"Node count increased: {report['nodes_before']} → {report['nodes_after']}"

    # Output model must be a valid ONNX graph
    optimized = onnx.load(OUTPUT_PATH_M2)
    onnx.checker.check_model(optimized)

    # Final verify pass
    original = onnx.load(MODEL_PATH)
    vreport = verify(original, optimized, n_samples=10)
    assert vreport.passed, f"Accuracy check failed: max_diff={vreport.max_diff}"

    print(f"\n✓ M2 test passed")
    print(f"  Nodes: {report['nodes_before']} → {report['nodes_after']}")
    print(f"  Size:  {report['size_before_mb']} MB → {report['size_after_mb']} MB")
    print(f"  Max diff: {vreport.max_diff:.2e}")


def test_m3_mobilenetv2():
    """Test M3 passes: eliminate_unused_initializers + eliminate_duplicate_constants"""
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}. Download from ONNX Model Zoo."

    original = onnx.load(MODEL_PATH)
    model = onnx.load(MODEL_PATH)

    init_before = len(original.graph.initializer)

    # Run M3 passes
    model = EliminateUnusedInitializers().run(model)
    model = EliminateDuplicateConstants().run(model)

    init_after = len(model.graph.initializer)

    # Verify accuracy preservation
    vreport = verify(original, model, n_samples=10)
    assert vreport.passed, f"Accuracy check failed: max_diff={vreport.max_diff}"
    assert vreport.max_diff < 1e-5

    # Output model must be valid
    onnx.checker.check_model(model)
    onnx.save(model, OUTPUT_PATH_M3)

    print(f"\n✓ M3 test passed")
    print(f"  Initializers: {init_before} → {init_after}")
    print(f"  Max diff: {vreport.max_diff:.2e}")


def test_all_passes():
    """Test all M2+M3 passes together through optimizer"""
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}. Download from ONNX Model Zoo."

    passes = [
        EliminateDeadNodes(),
        EliminateIdentityOps(),
        EliminateUnusedInitializers(),
        EliminateDuplicateConstants(),
    ]

    report = optimize(
        model_path=MODEL_PATH,
        output_path=OUTPUT_PATH_M3,
        passes=passes,
        verify_each_pass=True,
        n_verify_samples=10,
    )

    # Output model must be valid
    optimized = onnx.load(OUTPUT_PATH_M3)
    onnx.checker.check_model(optimized)

    # Final verify
    original = onnx.load(MODEL_PATH)
    vreport = verify(original, optimized, n_samples=10)
    assert vreport.passed

    print(f"\n✓ All passes test passed")
    print(f"  Nodes: {report['nodes_before']} → {report['nodes_after']}")
    print(f"  Size:  {report['size_before_mb']} MB → {report['size_after_mb']} MB")
    print(f"  Max diff: {vreport.max_diff:.2e}")


if __name__ == "__main__":
    test_m2_mobilenetv2()
    test_m3_mobilenetv2()
    test_all_passes()
