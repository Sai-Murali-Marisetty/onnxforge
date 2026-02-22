"""
Builds synthetic dirty ONNX models for testing simplify_shape_chains.
Run: python tests/toy_models/build_shape_chain_model.py
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def build_identity_reshape(output_path="tests/toy_models/shape_identity_reshape.onnx"):
    """
    Reshape where input shape == target shape → redundant, should be removed.

    X: [1, 4, 4]
    shape_const: [1, 4, 4]   ← constant, matches input shape exactly
    Reshape(X, [1,4,4]) → Y  ← does nothing

    Before: 2 nodes (Constant + Reshape)
    After:  0 nodes (X directly wires to Y)

    Note: with no nodes remaining, X IS Y after rewiring.
    """
    shape_val = np.array([1, 4, 4], dtype=np.int64)
    shape_tensor = numpy_helper.from_array(shape_val, name="shape_val")
    const_shape = helper.make_node(
        "Constant", [], ["shape"], value=shape_tensor, name="ConstShape"
    )
    reshape = helper.make_node("Reshape", ["X", "shape"], ["Y"], name="Reshape")

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 4])

    graph = helper.make_graph([const_shape, reshape], "identity_reshape", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    # Run shape inference so value_info is populated
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (Reshape with same shape — should be removed)")


def build_real_reshape(output_path="tests/toy_models/shape_real_reshape.onnx"):
    """
    Reshape that actually changes shape — must NOT be removed.

    X: [1, 16]
    Reshape(X, [4, 4]) → Y: [4, 4]

    Before: 2 nodes (Constant + Reshape)
    After:  2 nodes (untouched — reshape actually does something)
    """
    shape_val = np.array([4, 4], dtype=np.int64)
    shape_tensor = numpy_helper.from_array(shape_val, name="shape_val")
    const_shape = helper.make_node(
        "Constant", [], ["shape"], value=shape_tensor, name="ConstShape"
    )
    reshape = helper.make_node("Reshape", ["X", "shape"], ["Y"], name="Reshape")

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 4])

    graph = helper.make_graph([const_shape, reshape], "real_reshape", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (Reshape changes shape — must survive)")


def build_dead_shape_node(output_path="tests/toy_models/shape_dead_shape.onnx"):
    """
    A Shape node whose output is never consumed (dead after folding cleaned up consumers).

    X → Shape → [dead, no consumer]
    X → Relu  → Y

    Before: 2 nodes (Shape + Relu)
    After:  1 node (Shape removed, Relu survives)
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    shape_node = helper.make_node("Shape", ["X"], ["shape_out"], name="ShapeNode")
    relu       = helper.make_node("Relu",  ["X"], ["Y"],         name="Relu")

    graph = helper.make_graph([shape_node, relu], "dead_shape", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (Shape node with no consumer — should be removed)")


if __name__ == "__main__":
    os.makedirs("tests/toy_models", exist_ok=True)
    print("Building toy models for M6...\n")
    build_identity_reshape()
    build_real_reshape()
    build_dead_shape_node()
    print("\nDone. All models saved to tests/toy_models/")
