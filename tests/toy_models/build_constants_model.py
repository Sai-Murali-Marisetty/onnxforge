"""
Builds synthetic dirty ONNX models for testing fold_constants.
Run: python tests/toy_models/build_constants_model.py
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def build_simple_add(output_path="tests/toy_models/constants_add.onnx"):
    """
    Two Constant tensors fed into Add.
    All inputs are constants → Add collapses to one Constant.

    Before: Const_A + Const_B → Add → output
            3 nodes (2 Constant + 1 Add)
    After:  Constant([2,4,6]) → output
            1 node

    We can assert the output value exactly: [1+1, 2+2, 3+3] = [2, 4, 6]
    """
    a_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    a_tensor = numpy_helper.from_array(a_val, name="a_val")
    b_tensor = numpy_helper.from_array(b_val, name="b_val")

    const_a = helper.make_node("Constant", [], ["A"], value=a_tensor, name="ConstA")
    const_b = helper.make_node("Constant", [], ["B"], value=b_tensor, name="ConstB")
    add     = helper.make_node("Add", ["A", "B"], ["Y"], name="Add")

    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])
    graph = helper.make_graph([const_a, const_b, add], "simple_add", [], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (3 nodes → expect 1 after fold)")


def build_chain_fold(output_path="tests/toy_models/constants_chain.onnx"):
    """
    A chain of constant ops simulating a positional encoding computation.
    base * scale → unsqueeze

    Before: 4 nodes (2 Constant + Mul + Unsqueeze)
    After:  1 node with pre-computed [[0,2,4,6]]
    """
    base_val  = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    scale_val = np.array([2.0], dtype=np.float32)

    base_tensor  = numpy_helper.from_array(base_val,  name="base_val")
    scale_tensor = numpy_helper.from_array(scale_val, name="scale_val")

    const_base  = helper.make_node("Constant", [], ["base"],  value=base_tensor,  name="ConstBase")
    const_scale = helper.make_node("Constant", [], ["scale"], value=scale_tensor, name="ConstScale")
    mul         = helper.make_node("Mul", ["base", "scale"], ["scaled"], name="Mul")
    
    # Unsqueeze with axes as input (opset 13+)
    axes_val = np.array([0], dtype=np.int64)
    axes_tensor = numpy_helper.from_array(axes_val, name="axes_val")
    const_axes = helper.make_node("Constant", [], ["axes"], value=axes_tensor, name="ConstAxes")
    unsqueeze = helper.make_node("Unsqueeze", ["scaled", "axes"], ["Y"], name="Unsqueeze")

    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph(
        [const_base, const_scale, mul, const_axes, unsqueeze],
        "chain_fold", [], [Y]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (5 nodes → expect 1 after fold)")


def build_mixed_model(output_path="tests/toy_models/constants_mixed.onnx"):
    """
    Model with both constant and runtime (dynamic) subgraphs.
    Only the constant part should fold — the dynamic Add must survive.

    Constant(bias) → [pre-computable]
    runtime_input  → Add(runtime_input, bias) → output

    Key test: runtime Add node MUST survive. Only dead constant plumbing removed.
    """
    bias_val = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    bias_tensor = numpy_helper.from_array(bias_val, name="bias_val")

    X    = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])
    Y    = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])

    const_bias = helper.make_node("Constant", [], ["bias"], value=bias_tensor, name="ConstBias")
    add        = helper.make_node("Add", ["X", "bias"], ["Y"], name="Add")

    graph = helper.make_graph([const_bias, add], "mixed_model", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (bias is constant-fed Add — Add must survive)")


def build_no_fold_model(output_path="tests/toy_models/constants_no_fold.onnx"):
    """
    Model with zero constant subgraphs — all ops depend on runtime input.
    Pass should do nothing.

    X → Relu → Y
    Expected: 1 node → 1 node (untouched)
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])

    relu = helper.make_node("Relu", ["X"], ["Y"], name="Relu")

    graph = helper.make_graph([relu], "no_fold", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Built: {output_path}  (no constants — pass does nothing)")


if __name__ == "__main__":
    os.makedirs("tests/toy_models", exist_ok=True)
    print("Building toy models for M5...\n")
    build_simple_add()
    build_chain_fold()
    build_mixed_model()
    build_no_fold_model()
    print("\nDone. All models saved to tests/toy_models/")
