"""
Experiment: Find the right approach for Conv+ReLU fusion that:
1. ORT can still run (verify.py continues to work)
2. Provides value to downstream converters (TFLite, CoreML)
3. Is provably correct numerically

Test four approaches. Document exactly which ones ORT accepts and what
the output difference is. This determines how we fix M8.
"""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper
import copy
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def make_conv_relu_model():
    W = np.ones((2, 1, 3, 3), dtype=np.float32)
    X    = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])
    Y    = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 3, 3])
    conv = helper.make_node("Conv", ["X", "W"], ["conv_out"], kernel_shape=[3, 3])
    relu = helper.make_node("Relu", ["conv_out"], ["Y"])
    graph = helper.make_graph([conv, relu], "conv_relu", [X], [Y])
    graph.initializer.append(numpy_helper.from_array(W, "W"))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def approach_a(model):
    """Remove Relu, add custom activation attribute to Conv."""
    m = copy.deepcopy(model)
    relu = next(n for n in m.graph.node if n.op_type == "Relu")
    conv = next(n for n in m.graph.node if n.op_type == "Conv")
    conv.attribute.append(helper.make_attribute("activation", "Relu"))
    conv.output[0] = relu.output[0]
    m.graph.node.remove(relu)
    return m


def approach_b(model):
    """Keep Relu, add annotation-only attribute to Conv. Graph unchanged."""
    m = copy.deepcopy(model)
    for node in m.graph.node:
        if node.op_type == "Conv":
            conv_out = node.output[0]
            for n2 in m.graph.node:
                if n2.op_type == "Relu" and n2.input[0] == conv_out:
                    node.attribute.append(
                        helper.make_attribute("activation_hint", "Relu")
                    )
    return m


def approach_c(model):
    """Replace Relu with Clip(0, inf) — semantically identical, better TFLite compat."""
    m = copy.deepcopy(model)
    relu = next(n for n in m.graph.node if n.op_type == "Relu")
    min_t = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "clip_min")
    max_t = numpy_helper.from_array(np.array(3.4e38, dtype=np.float32), "clip_max")
    m.graph.initializer.extend([min_t, max_t])
    clip = helper.make_node(
        "Clip",
        inputs=[relu.input[0], "clip_min", "clip_max"],
        outputs=[relu.output[0]],
    )
    idx = list(m.graph.node).index(relu)
    m.graph.node.remove(relu)
    m.graph.node.insert(idx, clip)
    return m


def approach_d(model):
    """
    Two-model strategy: keep original for ORT verification,
    produce a separate export-only version with Relu removed.
    This approach separates ORT verification from converter output.
    """
    # For ORT: model unchanged
    ort_model = copy.deepcopy(model)
    # For converter: Relu removed with annotation
    export_model = approach_a(model)
    return ort_model, export_model


def test_ort(model, label):
    try:
        sess = ort.InferenceSession(model.SerializeToString())
        inp = {"X": np.random.randn(1, 1, 5, 5).astype(np.float32)}
        out = sess.run(None, inp)
        return True, out[0]
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    print("Experiment 05 — Conv+ReLU ORT Compatibility\n")

    baseline = make_conv_relu_model()
    ok, base_out = test_ort(baseline, "baseline")
    print(f"Baseline ORT: {'✓' if ok else '✗'}\n")

    experiments = [
        ("A — Relu removed, custom attr", approach_a),
        ("B — Relu kept, annotation only", approach_b),
        ("C — Relu → Clip(0, inf)", approach_c),
    ]

    for label, fn in experiments:
        m = fn(baseline)
        ok, out_or_err = test_ort(m, label)
        print(f"Approach {label}:")
        print(f"  ORT accepts: {'✓' if ok else '✗'}")
        if ok:
            diff = float(np.max(np.abs(base_out - out_or_err)))
            print(f"  max_diff vs baseline: {diff:.2e}")
        else:
            print(f"  ORT error: {str(out_or_err)[:300]}")
        print()

    print("Approach D — Two-model strategy:")
    ort_m, export_m = approach_d(baseline)
    ok_ort, out_ort = test_ort(ort_m, "D-ort")
    ok_exp, out_exp = test_ort(export_m, "D-export")
    print(f"  ORT model accepts: {'✓' if ok_ort else '✗'}")
    print(f"  Export model ORT:  {'✓' if ok_exp else '✗'}")
    if ok_ort and ok_exp:
        diff = float(np.max(np.abs(out_ort - out_exp)))
        print(f"  Diff between versions: {diff:.2e}")
