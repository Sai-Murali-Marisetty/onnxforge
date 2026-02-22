"""
Toy models for cleanup_attention pass.
Run: python tests/toy_models/build_attention_model.py
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def _add_init(graph, name, array):
    graph.initializer.append(
        numpy_helper.from_array(array.astype(np.float32 if array.dtype == np.float64 else array.dtype), name=name)
    )


def build_consecutive_reshape(output_path="tests/toy_models/consecutive_reshape.onnx"):
    """
    Reshape → Reshape (consecutive, mergeable)
    Before: 2 Reshape nodes
    After:  1 Reshape node
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 12, 64])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 768])
    
    shape1 = np.array([1, 12, 64], dtype=np.int64)
    shape2 = np.array([1, 768], dtype=np.int64)
    
    reshape1 = helper.make_node("Reshape", ["X", "shape1"], ["r1"], name="Reshape1")
    reshape2 = helper.make_node("Reshape", ["r1", "shape2"], ["Y"], name="Reshape2")
    
    graph = helper.make_graph([reshape1, reshape2], "consecutive_reshape", [X], [Y])
    _add_init(graph, "shape1", shape1)
    _add_init(graph, "shape2", shape2)
    
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (2 Reshape → expect 1 after cleanup)")


def build_identity_reshape(output_path="tests/toy_models/identity_reshape.onnx"):
    """
    Reshape to same shape (identity, removable in attention context)
    Input: [1, 12, 64] → Reshape → [1, 12, 64] → MatMul
    Before: Reshape + MatMul
    After:  MatMul only
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 12, 64])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 12, 32])
    
    shape1 = np.array([1, 12, 64], dtype=np.int64)
    W = np.random.randn(64, 32).astype(np.float32)
    
    reshape1 = helper.make_node("Reshape", ["X", "shape1"], ["r1"], name="Reshape1")
    matmul = helper.make_node("MatMul", ["r1", "W"], ["Y"], name="MatMul")
    
    graph = helper.make_graph([reshape1, matmul], "identity_reshape", [X], [Y])
    _add_init(graph, "shape1", shape1)
    _add_init(graph, "W", W)
    
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (identity Reshape → expect removal)")


def build_branching_reshape(output_path="tests/toy_models/branching_reshape.onnx"):
    """
    Reshape with multiple consumers — should NOT be removed.
    Before: 4 nodes (Reshape with 2 consumers)
    After:  4 nodes (unchanged)
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 768])
    Y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 768])
    Y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, 768])
    
    shape1 = np.array([1, 12, 64], dtype=np.int64)
    shape2 = np.array([1, 768], dtype=np.int64)
    
    reshape1 = helper.make_node("Reshape", ["X", "shape1"], ["r1"], name="Reshape1")
    # r1 has two consumers: Relu and Sigmoid
    relu = helper.make_node("Relu", ["r1"], ["relu_out"], name="Relu")
    sigmoid = helper.make_node("Sigmoid", ["r1"], ["sig_out"], name="Sigmoid")
    # Final reshapes back
    reshape2 = helper.make_node("Reshape", ["relu_out", "shape2"], ["Y1"], name="Reshape2")
    reshape3 = helper.make_node("Reshape", ["sig_out", "shape2"], ["Y2"], name="Reshape3")
    
    graph = helper.make_graph([reshape1, relu, sigmoid, reshape2, reshape3], 
                              "branching_reshape", [X], [Y1, Y2])
    _add_init(graph, "shape1", shape1)
    _add_init(graph, "shape2", shape2)
    
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (branching — 5 nodes stay 5)")


if __name__ == "__main__":
    os.makedirs("tests/toy_models", exist_ok=True)
    print("Building Attention cleanup toy models for M9...\n")
    build_consecutive_reshape()
    build_identity_reshape()
    build_branching_reshape()
    print("\nDone.")
