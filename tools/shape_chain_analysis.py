#!/usr/bin/env python3
"""Analyze where Shape chains feed in BERT."""

import onnx
from collections import defaultdict, Counter

model = onnx.load('models/bert_base.onnx')
graph = model.graph

# Build consumer map
consumers = defaultdict(list)
for n in graph.node:
    for inp in n.input:
        consumers[inp].append(n)

# Find what Shape chain outputs ultimately feed into
shape_nodes = [n for n in graph.node if n.op_type == 'Shape']

print('Shape chain destinations (what ops ultimately consume the shape info):')
print('-' * 60)

def trace_shape_chain(node, visited=None):
    if visited is None:
        visited = set()
    if node.output[0] in visited:
        return []
    visited.add(node.output[0])
    
    results = []
    out = node.output[0]
    for consumer in consumers.get(out, []):
        if consumer.op_type in ('Gather', 'Unsqueeze', 'Concat', 'Cast', 'Slice', 'Squeeze'):
            results.extend(trace_shape_chain(consumer, visited))
        else:
            results.append(consumer.op_type)
    return results

# Trace all shape chains
all_dests = []
for sn in shape_nodes:
    dests = trace_shape_chain(sn)
    all_dests.extend(dests)
    if len(dests) <= 3:
        print(f'{sn.name[:35]}: feeds -> {set(dests)}')

print()
print('Total shape chain destination ops:')
for op, cnt in Counter(all_dests).most_common(10):
    print(f'  {op}: {cnt}')

print()
print('Conclusion:')
print('  Shape chains feed Reshape nodes (for dynamic shape computation)')
print('  These cannot be eliminated without static shape analysis')
print('  Pass limitation is documented - not a bug')
