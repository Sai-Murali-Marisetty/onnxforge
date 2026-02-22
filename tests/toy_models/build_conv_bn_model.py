"""
Builds synthetic dirty ONNX models for testing fuse_conv_batchnorm.
Run: python tests/toy_models/build_conv_bn_model.py
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def _add_initializer(graph, name, array):
    tensor = numpy_helper.from_array(array.astype(np.float32), name=name)
    graph.initializer.append(tensor)


def build_conv_bn_pair(output_path="tests/toy_models/conv_bn_pair.onnx"):
    """
    Single Conv → BatchNorm pair.
    Conv: 1 input channel, 2 output channels, 3x3 kernel
    BN:   2 channels

    Before: 2 nodes (Conv + BN)
    After:  1 node  (Conv with fused weights)

    We set known weight values so we can assert the fused result exactly.
    """
    # Conv weights: [out=2, in=1, kH=3, kW=3] — all ones for easy math
    W = np.ones((2, 1, 3, 3), dtype=np.float32)
    # Conv bias: zeros
    b = np.zeros(2, dtype=np.float32)

    # BN params: gamma=1, beta=0, mean=0, var=1 → scale=1/(sqrt(1+eps))≈1
    # This means fused weights ≈ W * 1 = W (easy to verify)
    gamma = np.array([1.0, 1.0], dtype=np.float32)
    beta  = np.array([0.0, 0.0], dtype=np.float32)
    mean  = np.array([0.0, 0.0], dtype=np.float32)
    var   = np.array([1.0, 1.0], dtype=np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 3, 3])

    conv = helper.make_node(
        "Conv", ["X", "W", "b"], ["conv_out"],
        kernel_shape=[3, 3], name="Conv"
    )
    bn = helper.make_node(
        "BatchNormalization",
        ["conv_out", "gamma", "beta", "mean", "var"],
        ["Y"],
        epsilon=1e-5, name="BN"
    )

    graph = helper.make_graph([conv, bn], "conv_bn_pair", [X], [Y])
    _add_initializer(graph, "W",     W)
    _add_initializer(graph, "b",     b)
    _add_initializer(graph, "gamma", gamma)
    _add_initializer(graph, "beta",  beta)
    _add_initializer(graph, "mean",  mean)
    _add_initializer(graph, "var",   var)

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (Conv+BN → 2 nodes, expect 1 after fuse)")


def build_conv_no_bn(output_path="tests/toy_models/conv_no_bn.onnx"):
    """
    Conv with no following BN — pass should do nothing.
    Before: 1 node (Conv)
    After:  1 node (untouched)
    """
    W = np.ones((2, 1, 3, 3), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 3, 3])

    conv = helper.make_node("Conv", ["X", "W", "b"], ["Y"],
                            kernel_shape=[3, 3], name="Conv")

    graph = helper.make_graph([conv], "conv_no_bn", [X], [Y])
    _add_initializer(graph, "W", W)
    _add_initializer(graph, "b", b)

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (Conv only — pass does nothing)")


def build_two_conv_bn_pairs(output_path="tests/toy_models/conv_bn_double.onnx"):
    """
    Two sequential Conv+BN pairs.
    Before: 4 nodes (Conv BN Conv BN)
    After:  2 nodes (Conv Conv — both BNs fused)
    """
    W1 = np.ones((4, 1, 3, 3), dtype=np.float32)
    b1 = np.zeros(4, dtype=np.float32)
    W2 = np.ones((4, 4, 1, 1), dtype=np.float32)
    b2 = np.zeros(4, dtype=np.float32)

    # BN params for both (identity-ish: gamma=1, beta=0, mean=0, var=1)
    gamma1 = np.ones(4,  dtype=np.float32)
    beta1  = np.zeros(4, dtype=np.float32)
    mean1  = np.zeros(4, dtype=np.float32)
    var1   = np.ones(4,  dtype=np.float32)

    gamma2 = np.ones(4,  dtype=np.float32) * 2   # non-trivial scale
    beta2  = np.ones(4,  dtype=np.float32)        # non-zero bias
    mean2  = np.ones(4,  dtype=np.float32) * 0.5
    var2   = np.ones(4,  dtype=np.float32)

    X = helper.make_tensor_value_info("X",  TensorProto.FLOAT, [1, 1, 5, 5])
    Y = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, [1, 4, 3, 3])

    conv1 = helper.make_node("Conv", ["X",       "W1", "b1"], ["c1"], kernel_shape=[3, 3], name="Conv1")
    bn1   = helper.make_node("BatchNormalization", ["c1", "g1", "bt1", "m1", "v1"], ["bn1_out"], epsilon=1e-5, name="BN1")
    conv2 = helper.make_node("Conv", ["bn1_out", "W2", "b2"], ["c2"], kernel_shape=[1, 1], name="Conv2")
    bn2   = helper.make_node("BatchNormalization", ["c2", "g2", "bt2", "m2", "v2"], ["Y"],      epsilon=1e-5, name="BN2")

    graph = helper.make_graph([conv1, bn1, conv2, bn2], "double_conv_bn", [X], [Y])
    for name, arr in [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2),
                      ("g1", gamma1), ("bt1", beta1), ("m1", mean1), ("v1", var1),
                      ("g2", gamma2), ("bt2", beta2), ("m2", mean2), ("v2", var2)]:
        _add_initializer(graph, name, arr)

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (4 nodes → expect 2 after fuse)")


if __name__ == "__main__":
    os.makedirs("tests/toy_models", exist_ok=True)
    print("Building toy models for M7...\n")
    build_conv_bn_pair()
    build_conv_no_bn()
    build_two_conv_bn_pairs()
    print("\nDone. All models saved to tests/toy_models/")
