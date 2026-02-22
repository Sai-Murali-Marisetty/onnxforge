"""
Quick test for Clip vs Relu numerical equivalence
"""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper
import copy

def make_conv_relu_model():
    W = np.ones((2, 1, 3, 3), dtype=np.float32)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 5, 5])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 3, 3])
    conv = helper.make_node('Conv', ['X', 'W'], ['conv_out'], kernel_shape=[3, 3])
    relu = helper.make_node('Relu', ['conv_out'], ['Y'])
    graph = helper.make_graph([conv, relu], 'conv_relu', [X], [Y])
    graph.initializer.append(numpy_helper.from_array(W, 'W'))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    return model

def approach_c(model):
    m = copy.deepcopy(model)
    relu = next(n for n in m.graph.node if n.op_type == 'Relu')
    min_t = numpy_helper.from_array(np.array(0.0, dtype=np.float32), 'clip_min')
    max_t = numpy_helper.from_array(np.array(3.4e38, dtype=np.float32), 'clip_max')
    m.graph.initializer.extend([min_t, max_t])
    clip = helper.make_node('Clip', inputs=[relu.input[0], 'clip_min', 'clip_max'], outputs=[relu.output[0]])
    idx = list(m.graph.node).index(relu)
    m.graph.node.remove(relu)
    m.graph.node.insert(idx, clip)
    return m

if __name__ == "__main__":
    baseline = make_conv_relu_model()
    clip_model = approach_c(baseline)

    np.random.seed(42)
    inp = {'X': np.random.randn(1, 1, 5, 5).astype(np.float32)}

    sess1 = ort.InferenceSession(baseline.SerializeToString())
    sess2 = ort.InferenceSession(clip_model.SerializeToString())

    out1 = sess1.run(None, inp)[0]
    out2 = sess2.run(None, inp)[0]

    print('Same seed test:')
    print(f'  Baseline shape: {out1.shape}')
    print(f'  Clip shape: {out2.shape}')
    print(f'  Max diff: {np.max(np.abs(out1 - out2)):.2e}')
    print(f'  Are equal: {np.allclose(out1, out2)}')
