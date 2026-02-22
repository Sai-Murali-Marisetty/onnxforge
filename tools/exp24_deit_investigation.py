#!/usr/bin/env python3
"""
Exp 24 — DeiT Regression Investigation + DeiT vs BERT Identity Ops

Why did DeiT get -0.4% latency from 13.5% node reduction?
Why does DeiT have 122 identity ops but BERT has 0?
"""

import os
import onnx
from collections import Counter, defaultdict

def count_identity_ops(model_path, model_name):
    """Count Identity ops and trace their sources."""
    model = onnx.load(model_path)
    graph = model.graph
    
    print(f"\n{'='*70}")
    print(f"Identity Op Analysis: {model_name}")
    print('='*70)
    
    identity_nodes = [n for n in graph.node if n.op_type == 'Identity']
    print(f"Identity nodes: {len(identity_nodes)}")
    
    if len(identity_nodes) == 0:
        print("  No Identity nodes found")
        return 0
    
    # Build producer map
    producers = {out: n for n in graph.node for out in n.output}
    
    # Trace what produces Identity inputs
    producer_types = Counter()
    for node in identity_nodes:
        inp = node.input[0]
        if inp in producers:
            producer_types[producers[inp].op_type] += 1
        else:
            # Check if it's a graph input or initializer
            init_names = {i.name for i in graph.initializer}
            inp_names = {i.name for i in graph.input}
            if inp in init_names:
                producer_types['<initializer>'] += 1
            elif inp in inp_names:
                producer_types['<graph_input>'] += 1
            else:
                producer_types['<unknown>'] += 1
    
    print(f"\nIdentity input sources:")
    for op, count in producer_types.most_common():
        print(f"  {op}: {count}")
    
    # Trace what consumes Identity outputs
    consumers = defaultdict(list)
    for n in graph.node:
        for inp in n.input:
            consumers[inp].append(n)
    
    consumer_types = Counter()
    for node in identity_nodes:
        out = node.output[0]
        for consumer in consumers.get(out, []):
            consumer_types[consumer.op_type] += 1
    
    print(f"\nIdentity output consumers:")
    for op, count in consumer_types.most_common(10):
        print(f"  {op}: {count}")
    
    return len(identity_nodes)

def compare_op_histograms(path1, name1, path2, name2):
    """Compare op type distributions between two models."""
    model1 = onnx.load(path1)
    model2 = onnx.load(path2)
    
    ops1 = Counter(n.op_type for n in model1.graph.node)
    ops2 = Counter(n.op_type for n in model2.graph.node)
    
    print(f"\n{'='*70}")
    print(f"Op Histogram Comparison: {name1} vs {name2}")
    print('='*70)
    
    all_ops = set(ops1.keys()) | set(ops2.keys())
    
    print(f"{'Op':<25} {name1:>10} {name2:>10} {'Diff':>10}")
    print("-" * 55)
    
    diffs = []
    for op in sorted(all_ops):
        c1 = ops1.get(op, 0)
        c2 = ops2.get(op, 0)
        diff = c2 - c1
        if diff != 0:
            diffs.append((op, c1, c2, diff))
    
    # Sort by absolute difference
    for op, c1, c2, diff in sorted(diffs, key=lambda x: -abs(x[3])):
        sign = '+' if diff > 0 else ''
        print(f"{op:<25} {c1:>10} {c2:>10} {sign}{diff:>9}")

def analyze_deit_identity_pattern(model_path):
    """Deep dive into DeiT's Identity ops."""
    model = onnx.load(model_path)
    graph = model.graph
    
    print(f"\n{'='*70}")
    print("DeiT Identity Pattern Deep Dive")
    print('='*70)
    
    identity_nodes = [n for n in graph.node if n.op_type == 'Identity']
    
    # Find patterns: what's the typical subgraph around Identity?
    producers = {out: n for n in graph.node for out in n.output}
    consumers = defaultdict(list)
    for n in graph.node:
        for inp in n.input:
            consumers[inp].append(n)
    
    # Sample a few Identity nodes
    print("\nSample Identity node contexts:")
    print("-" * 60)
    
    for node in identity_nodes[:5]:
        inp = node.input[0]
        out = node.output[0]
        
        producer = producers.get(inp, None)
        consumer_list = consumers.get(out, [])
        
        prod_type = producer.op_type if producer else "<input>"
        cons_types = [c.op_type for c in consumer_list[:3]]
        
        print(f"  {prod_type} -> Identity -> {cons_types}")

def main():
    models_dir = "models"
    
    print("=" * 70)
    print("Exp 24: DeiT Regression & Identity Ops Investigation")
    print("=" * 70)
    print("""
Questions:
1. Why did DeiT get -0.4% latency from 13.5% node reduction?
2. Why does DeiT have 122 identity ops but BERT has 0?
""")
    
    # Count Identity ops in various models
    bert_identity = count_identity_ops(
        os.path.join(models_dir, "bert_base.onnx"), "BERT-base"
    )
    deit_identity = count_identity_ops(
        os.path.join(models_dir, "deit_small.onnx"), "DeiT-Small"
    )
    vit_identity = count_identity_ops(
        os.path.join(models_dir, "vit_base.onnx"), "ViT-Base"
    )
    
    # Compare BERT vs DeiT
    compare_op_histograms(
        os.path.join(models_dir, "bert_base.onnx"), "BERT",
        os.path.join(models_dir, "deit_small.onnx"), "DeiT"
    )
    
    # Deep dive into DeiT Identity pattern
    analyze_deit_identity_pattern(os.path.join(models_dir, "deit_small.onnx"))
    
    print(f"\n{'='*70}")
    print("FINDINGS")
    print('='*70)
    print(f"""
Identity op counts:
  BERT:  {bert_identity}
  DeiT:  {deit_identity}
  ViT:   {vit_identity}
  
Key question: Why does DeiT export with 122 Identity ops?

Hypothesis 1: HuggingFace export path
  - DeiT uses a different model class than BERT
  - The exporter may insert Identity ops at certain boundaries
  - These might be "no-op" markers for debugging

Hypothesis 2: Distillation artifacts  
  - DeiT = "Data-efficient Image Transformer" (trained with distillation)
  - The distillation head might leave Identity artifacts

Hypothesis 3: ORT optimization hints
  - Identity ops might serve as fusion boundaries
  - Removing them might disrupt ORT's internal graph optimization
  - This could explain the -0.4% latency regression

To test Hypothesis 3:
  - Run DeiT with ORT graph optimization DISABLED
  - Compare: onnxforge-optimized vs. original with ORT opt disabled
  - If onnxforge helps more when ORT can't optimize → they were competing
""")

if __name__ == "__main__":
    main()
