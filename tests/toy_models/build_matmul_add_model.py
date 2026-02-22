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
    np.random.seed(42)  # for reproducibility
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
    np.random.seed(42)
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
    np.random.seed(42)
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
