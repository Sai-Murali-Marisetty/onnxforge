"""
Toy models for fuse_conv_relu.
Run: python tests/toy_models/build_conv_relu_model.py
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def _add_init(graph, name, array):
    graph.initializer.append(
        numpy_helper.from_array(array.astype(np.float32), name=name)
    )


def build_conv_relu(output_path="tests/toy_models/conv_relu.onnx"):
    """
    Conv → Relu → output
    Before: 2 nodes
    After:  1 Conv node with activation='Relu' attribute
    """
    W = np.ones((2, 1, 3, 3), dtype=np.float32)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 3, 3])

    conv = helper.make_node("Conv", ["X", "W"], ["conv_out"], kernel_shape=[3, 3], name="Conv")
    relu = helper.make_node("Relu", ["conv_out"], ["Y"], name="Relu")

    graph = helper.make_graph([conv, relu], "conv_relu", [X], [Y])
    _add_init(graph, "W", W)

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (2 nodes → expect 1 after fuse)")


def build_conv_no_relu(output_path="tests/toy_models/conv_no_relu.onnx"):
    """Conv without Relu — pass does nothing."""
    W = np.ones((2, 1, 3, 3), dtype=np.float32)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 3, 3])

    conv = helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[3, 3], name="Conv")

    graph = helper.make_graph([conv], "conv_no_relu", [X], [Y])
    _add_init(graph, "W", W)

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (Conv only — untouched)")


def build_conv_relu_conv(output_path="tests/toy_models/conv_relu_conv.onnx"):
    """
    Conv → Relu → Conv (Relu between two Convs).
    First Relu fuses into first Conv.
    Second Conv has no Relu — stays alone.
    Before: 3 nodes
    After:  2 nodes (first Conv has activation attr, second Conv untouched)
    """
    W1 = np.ones((4, 1, 3, 3), dtype=np.float32)
    W2 = np.ones((4, 4, 1, 1), dtype=np.float32)

    X  = helper.make_tensor_value_info("X",  TensorProto.FLOAT, [1, 1, 5, 5])
    Y  = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, [1, 4, 3, 3])

    conv1 = helper.make_node("Conv",  ["X",       "W1"], ["c1"],  kernel_shape=[3, 3], name="Conv1")
    relu  = helper.make_node("Relu",  ["c1"],            ["r1"],  name="Relu")
    conv2 = helper.make_node("Conv",  ["r1",      "W2"], ["Y"],   kernel_shape=[1, 1], name="Conv2")

    graph = helper.make_graph([conv1, relu, conv2], "conv_relu_conv", [X], [Y])
    _add_init(graph, "W1", W1)
    _add_init(graph, "W2", W2)

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (3 nodes → expect 2 after fuse)")


if __name__ == "__main__":
    os.makedirs("tests/toy_models", exist_ok=True)
    print("Building Conv+ReLU toy models for M8...\n")
    build_conv_relu()
    build_conv_no_relu()
    build_conv_relu_conv()
    print("\nDone.")
