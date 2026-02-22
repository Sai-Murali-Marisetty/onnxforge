"""
Inspect any ONNX model — full op inventory and pass relevance flags.
Usage: python tests/experiments/exp_08_graph_inspector.py models/yolov8n.onnx
       python tests/experiments/exp_08_graph_inspector.py  (runs all models)
"""
import sys
import os
import onnx
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def inspect(model_path):
    model     = onnx.load(model_path)
    graph     = model.graph
    op_counts = Counter(n.op_type for n in graph.node)
    nodes     = len(graph.node)
    size_mb   = model.ByteSize() / 1024 / 1024
    n_inits   = len(graph.initializer)
    opset     = next((o.version for o in model.opset_import
                      if o.domain in ('', 'ai.onnx')), '?')

    print(f"\n{'='*58}")
    print(f"  {model_path}")
    print(f"{'='*58}")
    print(f"  Nodes: {nodes}   Initializers: {n_inits}   "
          f"Size: {size_mb:.1f}MB   Opset: {opset}")
    print(f"\n  Op inventory:")
    for op, count in op_counts.most_common():
        bar = '█' * min(count, 50)
        print(f"    {op:<28} {count:>5}  {bar}")

    print(f"\n  Pass relevance:")
    checks = [
        ("M3 unused_inits",       f"{n_inits} initializers"),
        ("M4 transposes",         f"{op_counts.get('Transpose', 0)} Transpose"),
        ("M5 fold_constants",     f"{op_counts.get('Constant', 0)} Constant"),
        ("M6 shape_chains",       f"{op_counts.get('Reshape', 0)} Reshape  {op_counts.get('Shape', 0)} Shape"),
        ("M7 conv_bn",            f"{op_counts.get('Conv', 0)} Conv  {op_counts.get('BatchNormalization', 0)} BN"),
        ("M8 conv_relu",          f"{op_counts.get('Relu', 0)} Relu  {op_counts.get('Clip', 0)} Clip"),
        ("M8 matmul_add→gemm",    f"{op_counts.get('MatMul', 0)} MatMul  {op_counts.get('Add', 0)} Add  {op_counts.get('Gemm', 0)} Gemm"),
        ("M9 attention_cleanup",  f"{op_counts.get('Reshape', 0)} Reshape  {op_counts.get('Transpose', 0)} Transpose"),
    ]
    for label, detail in checks:
        print(f"    {label:<25} {detail}")

    print()
    return op_counts, nodes, size_mb


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else [
        "mobilenetv2-12.onnx",
        "models/yolov8n.onnx",
        "models/efficientnet-b0.onnx",
        "models/bert-base-uncased.onnx",
    ]
    for path in targets:
        if os.path.exists(path):
            inspect(path)
        else:
            print(f"\n⚠ Not found: {path}")
