"""Analyze BERT MatMul patterns."""
import onnx
from collections import defaultdict

model = onnx.load('models/bert_base.onnx')
graph = model.graph

# Build consumer map
input_to_consumers = defaultdict(list)
for node in graph.node:
    for inp in node.input:
        input_to_consumers[inp].append(node)

# Find MatMul -> Add pairs
matmul_add_pairs = []
for node in graph.node:
    if node.op_type == 'MatMul':
        for out in node.output:
            for consumer in input_to_consumers[out]:
                if consumer.op_type == 'Add':
                    matmul_add_pairs.append((node.name, consumer.name))

print(f'MatMul -> Add pairs found: {len(matmul_add_pairs)}')
for mm, add in matmul_add_pairs[:10]:
    print(f'  {mm[:50]} -> {add[:40]}')

# Check what ops consume MatMul outputs
print('\nOps consuming MatMul outputs:')
matmul_consumers = defaultdict(int)
for node in graph.node:
    if node.op_type == 'MatMul':
        for out in node.output:
            for consumer in input_to_consumers[out]:
                matmul_consumers[consumer.op_type] += 1

for op, count in sorted(matmul_consumers.items(), key=lambda x: -x[1]):
    print(f'  {op}: {count}')

# Analyze MatMul that DON'T go to Add
print('\nMatMul patterns not going to Add:')
matmul_non_add = []
for node in graph.node:
    if node.op_type == 'MatMul':
        for out in node.output:
            consumers = input_to_consumers[out]
            if not any(c.op_type == 'Add' for c in consumers):
                consumer_types = [c.op_type for c in consumers]
                matmul_non_add.append((node.name, consumer_types))

for mm, consumers in matmul_non_add[:10]:
    print(f'  {mm[:50]} -> {consumers}')
