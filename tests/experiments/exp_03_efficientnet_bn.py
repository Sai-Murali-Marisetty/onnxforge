"""
Experiment: Prove fuse_conv_batchnorm fires on EfficientNet-B0.
EfficientNet is dense with Conv+BN — all should fuse.
Also checks for unfused BN nodes and explains why.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import onnx
from verify import verify
from passes.fuse_conv_batchnorm import FuseConvBatchnorm


def count_ops(model, *ops):
    return {op: sum(1 for n in model.graph.node if n.op_type == op) for op in ops}


def find_unfused_bn(model):
    """Find BN nodes that don't immediately follow a Conv."""
    graph   = model.graph
    outputs = {out: node for node in graph.node for out in node.output}
    unfused = []
    for node in graph.node:
        if node.op_type == "BatchNormalization":
            producer = outputs.get(node.input[0])
            if producer is None or producer.op_type != "Conv":
                unfused.append((node.name, producer.op_type if producer else "graph_input"))
    return unfused


def run():
    path = "models/efficientnet-b0.onnx"
    
    if not os.path.exists(path):
        print(f"⚠ Not found: {path}")
        return
    
    print("Experiment 03 — EfficientNet-B0 Conv+BN Fusion\n")

    original = onnx.load(path)
    b4 = count_ops(original, "Conv", "BatchNormalization", "Relu", "Clip", "Sigmoid")
    size_b4 = original.ByteSize() / 1024 / 1024

    print("BEFORE:")
    for op, count in b4.items():
        print(f"  {op:<25} {count}")
    print(f"  Total nodes: {len(original.graph.node)}")
    print(f"  Size: {size_b4:.2f}MB")

    # Check for unfused BN before running (expect 0)
    unfused_before = find_unfused_bn(original)
    if unfused_before:
        print(f"\n  BN nodes not after Conv (pre-existing): {len(unfused_before)}")
        for name, prev in unfused_before[:3]:
            print(f"    BN '{name}' — producer is: {prev}")

    model     = onnx.load(path)
    optimized = FuseConvBatchnorm().run(model)
    af = count_ops(optimized, "Conv", "BatchNormalization", "Relu", "Clip", "Sigmoid")
    size_af = optimized.ByteSize() / 1024 / 1024

    print("\nAFTER:")
    for op in b4:
        delta = b4[op] - af[op]
        marker = f"  (-{delta})" if delta > 0 else ""
        print(f"  {op:<25} {af[op]}{marker}")
    print(f"  Total nodes: {len(optimized.graph.node)}")
    print(f"  Size: {size_af:.2f}MB  ({size_b4 - size_af:+.2f}MB)")

    # Check for remaining unfused BN
    unfused_after = find_unfused_bn(optimized)
    if unfused_after:
        print(f"\n  ⚠ {len(unfused_after)} BN nodes NOT fused:")
        for name, prev in unfused_after[:5]:
            print(f"    BN '{name}' — producer is: {prev}")

    report = verify(original, optimized, n_samples=5, tolerance=1e-4)
    print(f"\nAccuracy: max_diff={report.max_diff:.2e}  "
          f"{'✓' if report.passed else '✗ FAILED'}")

    if af["BatchNormalization"] == 0:
        print("\n✓ All BN nodes fused.")
    else:
        print(f"\n⚠ {af['BatchNormalization']} BN nodes remain — investigate above.")


if __name__ == "__main__":
    run()
