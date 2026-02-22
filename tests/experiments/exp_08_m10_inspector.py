#!/usr/bin/env python3
"""
M10 Experiment 08 — Graph Inspector (All Models)

Build complete op-frequency baseline table for all M10 models.
"""
import sys
import os
import onnx
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# All M10 models
M10_MODELS = [
    # Vision models (existing)
    "mobilenetv2-12.onnx",
    "models/efficientnet-b0.onnx",
    # Vision models (new)
    "models/resnet50.onnx",
    "models/mobilenetv3_small.onnx",
    # Transformer models
    "models/bert_base.onnx",
    "models/distilbert_base.onnx",
    "models/roberta_base.onnx",
    # Audio models
    "models/whisper_tiny_encoder.onnx",
]

def inspect(model_path):
    """Inspect a single model and return key stats."""
    model = onnx.load(model_path)
    graph = model.graph
    op_counts = Counter(n.op_type for n in graph.node)
    nodes = len(graph.node)
    size_mb = model.ByteSize() / 1024 / 1024
    n_inits = len(graph.initializer)
    opset = next((o.version for o in model.opset_import
                  if o.domain in ('', 'ai.onnx')), '?')
    
    return {
        'path': model_path,
        'nodes': nodes,
        'inits': n_inits,
        'size_mb': size_mb,
        'opset': opset,
        'op_counts': op_counts,
    }

def print_model_report(info):
    """Print detailed report for one model."""
    print(f"\n{'='*70}")
    print(f"  {info['path']}")
    print(f"{'='*70}")
    print(f"  Nodes: {info['nodes']}   Initializers: {info['inits']}   "
          f"Size: {info['size_mb']:.1f}MB   Opset: {info['opset']}")
    
    print(f"\n  Op inventory (top 20):")
    for op, count in info['op_counts'].most_common(20):
        bar = '█' * min(count, 40)
        print(f"    {op:<28} {count:>5}  {bar}")
    
    op = info['op_counts']
    print(f"\n  Pass relevance:")
    checks = [
        ("M2 dead_nodes/identity",   f"Identity={op.get('Identity', 0)}"),
        ("M3 unused_inits",          f"{info['inits']} initializers"),
        ("M4 transposes",            f"Transpose={op.get('Transpose', 0)}"),
        ("M5 fold_constants",        f"Constant={op.get('Constant', 0)}"),
        ("M6 shape_chains",          f"Reshape={op.get('Reshape', 0)} Shape={op.get('Shape', 0)}"),
        ("M7 conv_bn",               f"Conv={op.get('Conv', 0)} BN={op.get('BatchNormalization', 0)}"),
        ("M8 conv_relu",             f"Relu={op.get('Relu', 0)} Clip={op.get('Clip', 0)}"),
        ("M8 matmul_add→gemm",       f"MatMul={op.get('MatMul', 0)} Add={op.get('Add', 0)} Gemm={op.get('Gemm', 0)}"),
        ("M9 attention",             f"Softmax={op.get('Softmax', 0)} LayerNorm={op.get('LayerNormalization', 0)}"),
    ]
    for label, detail in checks:
        print(f"    {label:<25} {detail}")

def print_summary_table(all_info):
    """Print the 12-model summary table."""
    print("\n" + "="*120)
    print("M10 OP FREQUENCY SUMMARY TABLE")
    print("="*120)
    
    # Header
    print(f"\n{'Model':<30} | {'Nodes':>6} | {'Conv':>5} | {'BN':>4} | {'Relu':>5} | "
          f"{'Clip':>5} | {'MatMul':>6} | {'Transp':>6} | {'Reshape':>7} | {'Softmax':>7} | {'LayerN':>6}")
    print("-"*120)
    
    for info in all_info:
        name = os.path.basename(info['path']).replace('.onnx', '')[:28]
        op = info['op_counts']
        print(f"{name:<30} | {info['nodes']:>6} | {op.get('Conv', 0):>5} | "
              f"{op.get('BatchNormalization', 0):>4} | {op.get('Relu', 0):>5} | "
              f"{op.get('Clip', 0):>5} | {op.get('MatMul', 0):>6} | "
              f"{op.get('Transpose', 0):>6} | {op.get('Reshape', 0):>7} | "
              f"{op.get('Softmax', 0):>7} | {op.get('LayerNormalization', 0):>6}")

def main():
    print("="*70)
    print("M10 Experiment 08 — Graph Inspector (All Models)")
    print("="*70)
    
    all_info = []
    
    for model_path in M10_MODELS:
        if os.path.exists(model_path):
            info = inspect(model_path)
            print_model_report(info)
            all_info.append(info)
        else:
            print(f"\n⚠ Not found: {model_path}")
    
    # Print summary table
    if all_info:
        print_summary_table(all_info)
    
    print("\n" + "="*70)
    print("Graph inspection complete!")
    print("="*70)
    
    return all_info

if __name__ == "__main__":
    main()
