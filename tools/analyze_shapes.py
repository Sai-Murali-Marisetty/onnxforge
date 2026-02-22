"""Analyze MatMul input shapes after shape inference."""
import onnx
from onnx import shape_inference

model = onnx.load('models/bert_base.onnx')
model = shape_inference.infer_shapes(model)

# Build shape map
shape_map = {}
for vi in model.graph.value_info:
    if vi.type.HasField('tensor_type'):
        shape = [d.dim_value if d.HasField('dim_value') else '?' for d in vi.type.tensor_type.shape.dim]
        shape_map[vi.name] = shape

# Check MatMul input shapes
print("Example MatMul (query projection):")
for node in model.graph.node:
    if node.op_type == 'MatMul' and 'query' in node.name:
        inp_a, inp_b = node.input
        shape_a = shape_map.get(inp_a, 'unknown')
        shape_b = shape_map.get(inp_b, 'unknown')
        print(f'  {node.name[:50]}:')
        print(f'    Input A: {shape_a}')
        print(f'    Input B: {shape_b}')
        break

# Count by rank
print("\nMatMul input A ranks:")
ranks = {}
for node in model.graph.node:
    if node.op_type == 'MatMul':
        inp_a = node.input[0]
        shape_a = shape_map.get(inp_a, [])
        rank = len(shape_a) if shape_a else 'unknown'
        ranks[rank] = ranks.get(rank, 0) + 1

for rank, count in sorted(ranks.items()):
    print(f'  Rank {rank}: {count}')

# Show some 3D examples
print("\n3D MatMul examples:")
count = 0
for node in model.graph.node:
    if node.op_type == 'MatMul':
        inp_a = node.input[0]
        shape_a = shape_map.get(inp_a, [])
        if len(shape_a) == 3:
            print(f'  {node.name[:50]}: input shape {shape_a}')
            count += 1
            if count >= 5:
                break
