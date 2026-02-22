"""Debug the fuse_matmul_add_3d pass."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np

# Create test model
weight = np.random.randn(64, 128).astype(np.float32)
bias = np.random.randn(64).astype(np.float32)

transpose_node = helper.make_node('Transpose', ['weight'], ['weight_t'], perm=[1, 0], name='transpose')
matmul_node = helper.make_node('MatMul', ['input', 'weight_t'], ['matmul_out'], name='matmul')
add_node = helper.make_node('Add', ['matmul_out', 'bias'], ['output'], name='add')

graph = helper.make_graph(
    [transpose_node, matmul_node, add_node],
    'test',
    [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 128])],
    [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 64])],
    [numpy_helper.from_array(weight, 'weight'), numpy_helper.from_array(bias, 'bias')]
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 14)])
print(f'Before: {len(model.graph.node)} nodes')
print('Nodes:', [n.op_type for n in model.graph.node])

# Check consumer map
from collections import defaultdict
consumer_map = defaultdict(list)
for node in model.graph.node:
    for inp in node.input:
        consumer_map[inp].append(node)

print(f"Consumers of 'weight_t': {[c.name for c in consumer_map.get('weight_t', [])]}")

# Check the pass logic
from passes.fuse_matmul_add_3d import FuseMatmulAdd3d, _build_consumer_map, _is_initializer

print(f"Is 'weight' an initializer? {_is_initializer(model.graph, 'weight')}")

# Now run the pass
fuse = FuseMatmulAdd3d()
optimized = fuse.run(model)

print(f'After: {len(optimized.graph.node)} nodes')
print('Nodes:', [n.op_type for n in optimized.graph.node])
print('Initializers:', [i.name for i in optimized.graph.initializer])
