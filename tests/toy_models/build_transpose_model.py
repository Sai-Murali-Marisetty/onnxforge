"""
Builds synthetic dirty ONNX models for testing eliminate_redundant_transposes.
Run directly to regenerate: python tests/toy_models/build_transpose_model.py
"""
import os
import onnx
from onnx import helper, TensorProto


def build_cancelling_pair(output_path="tests/toy_models/transpose_cancelling.onnx"):
    """
    Two Transposes that cancel each other (compose to identity).
        input  (1,3,4,4) NCHW
        → Transpose(perm=[0,2,3,1])   NCHW → NHWC
        → Transpose(perm=[0,3,1,2])   NHWC → NCHW  ← cancels first
        → output (1,3,4,4)
    Expected: 2 nodes → 0 nodes
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])

    t1 = helper.make_node("Transpose", ["X"],   ["mid"], perm=[0, 2, 3, 1], name="T1")
    t2 = helper.make_node("Transpose", ["mid"], ["Y"],   perm=[0, 3, 1, 2], name="T2")

    graph = helper.make_graph([t1, t2], "cancelling_pair", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (2 nodes → expect 0 after pass)")


def build_mergeable_chain(output_path="tests/toy_models/transpose_mergeable.onnx"):
    """
    Two Transposes that merge into one (compose to non-identity).
        input  (1,4,8,8)
        → Transpose(perm=[0,2,1,3])
        → Transpose(perm=[0,1,3,2])
        → output
    Composed perm: [0,3,1,2]  → not identity, 2 nodes become 1
    Expected: 2 nodes → 1 node
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 8, 8])
    # After perm=[0,2,1,3]: shape becomes [1,8,4,8]
    # After perm=[0,1,3,2]: shape becomes [1,8,8,4]
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 8, 4])

    t1 = helper.make_node("Transpose", ["X"],   ["mid"], perm=[0, 2, 1, 3], name="T1")
    t2 = helper.make_node("Transpose", ["mid"], ["Y"],   perm=[0, 1, 3, 2], name="T2")

    graph = helper.make_graph([t1, t2], "mergeable_chain", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (2 nodes → expect 1 after pass)")


def build_clean_model(output_path="tests/toy_models/transpose_clean.onnx"):
    """
    Single Transpose — nothing to eliminate.
    Pass should do nothing.
    Expected: 1 node → 1 node
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 4, 3])

    t1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="T1")

    graph = helper.make_graph([t1], "clean_model", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (1 node → expect 1 after pass)")


def build_triple_chain(output_path="tests/toy_models/transpose_triple.onnx"):
    """
    Three consecutive Transposes — first two cancel, third survives.
        T1: [0,2,3,1]
        T2: [0,3,1,2]  ← cancels T1
        T3: [0,2,3,1]  ← survives alone
    Expected: 3 nodes → 1 node
    """
    X  = helper.make_tensor_value_info("X",  TensorProto.FLOAT, [1, 3, 4, 4])
    Y  = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, [1, 4, 4, 3])

    t1 = helper.make_node("Transpose", ["X"],    ["mid1"], perm=[0, 2, 3, 1], name="T1")
    t2 = helper.make_node("Transpose", ["mid1"], ["mid2"], perm=[0, 3, 1, 2], name="T2")
    t3 = helper.make_node("Transpose", ["mid2"], ["Y"],    perm=[0, 2, 3, 1], name="T3")

    graph = helper.make_graph([t1, t2, t3], "triple_chain", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (3 nodes → expect 1 after pass)")


if __name__ == "__main__":
    os.makedirs("tests/toy_models", exist_ok=True)
    print("Building toy models for M4...\n")
    build_cancelling_pair()
    build_mergeable_chain()
    build_clean_model()
    build_triple_chain()
    print("\nDone. All models saved to tests/toy_models/")
