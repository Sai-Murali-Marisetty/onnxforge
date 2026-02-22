"""Debug why FuseMatmulAdd3d isn't finding candidates."""
import onnx
from onnx import shape_inference
from collections import defaultdict

model = onnx.load('models/bert_base.onnx')
print(f'Before shape inference: {len(model.graph.value_info)} value_info')

model = shape_inference.infer_shapes(model)
print(f'After shape inference: {len(model.graph.value_info)} value_info')

graph = model.graph

# Build value_info dict
value_info_dict = {}
for vi in graph.value_info:
    if vi.type.HasField('tensor_type'):
        dims = vi.type.tensor_type.shape.dim
        if dims:
            shape = []
            for d in dims:
                if d.HasField('dim_value') and d.dim_value > 0:
                    shape.append(d.dim_value)
                else:
                    shape.append(None)
            value_info_dict[vi.name] = shape

# Build consumer map
consumer_map = defaultdict(list)
for node in graph.node:
    for inp in node.input:
        consumer_map[inp].append(node)

initializer_names = {init.name for init in graph.initializer}

def get_shape(name):
    if name in value_info_dict:
        return value_info_dict[name]
    for inp in graph.input:
        if inp.name == name:
            dims = inp.type.tensor_type.shape.dim
            if dims:
                return [d.dim_value if d.HasField('dim_value') and d.dim_value > 0 else None for d in dims]
    for init in graph.initializer:
        if init.name == name:
            return list(init.dims)
    return None

# Check each MatMul
print("\nChecking MatMul nodes:")
candidates = 0
for node in graph.node:
    if node.op_type != 'MatMul':
        continue
    
    matmul_output = node.output[0]
    input_a = node.input[0]
    input_b = node.input[1]
    
    shape_a = get_shape(input_a)
    shape_b = get_shape(input_b)
    
    # Check conditions
    is_3d = shape_a is not None and len(shape_a) == 3
    is_weight_2d = shape_b is not None and len(shape_b) == 2
    is_weight_init = input_b in initializer_names
    
    # Find Add consumer
    consumers = consumer_map.get(matmul_output, [])
    add_consumer = None
    for c in consumers:
        if c.op_type == 'Add':
            add_consumer = c
            break
    
    if 'query' in node.name or 'key' in node.name or 'value' in node.name:
        if '/layer.0/' in node.name:
            print(f"\n{node.name}:")
            print(f"  input_a shape: {shape_a} (is_3d={is_3d})")
            print(f"  input_b shape: {shape_b} (is_2d={is_weight_2d}, is_init={is_weight_init})")
            print(f"  has Add consumer: {add_consumer is not None}")
            if add_consumer:
                # Check bias
                bias_name = None
                for inp in add_consumer.input:
                    if inp != matmul_output:
                        bias_name = inp
                print(f"  bias name: {bias_name}")
                print(f"  bias is init: {bias_name in initializer_names if bias_name else False}")
    
    if is_3d and is_weight_2d and is_weight_init and add_consumer:
        candidates += 1

print(f"\nTotal candidates: {candidates}")
