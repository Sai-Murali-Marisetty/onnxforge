import onnx
import time
from typing import List
from passes.base_pass import BasePass
from verify import verify, AccuracyLossError

def count_nodes(model: onnx.ModelProto) -> int:
    return len(model.graph.node)

def model_size_mb(model: onnx.ModelProto) -> float:
    return model.ByteSize() / (1024 * 1024)

def optimize(
    model_path: str,
    output_path: str,
    passes: List[BasePass],
    verify_each_pass: bool = True,
    n_verify_samples: int = 5
) -> dict:
    """
    Load model, run passes in sequence, verify, save.
    Returns a report dict with before/after stats.
    """
    print(f"\nLoading: {model_path}")
    original = onnx.load(model_path)
    model = onnx.load(model_path)  # working copy

    nodes_before = count_nodes(model)
    size_before  = model_size_mb(model)
    passes_applied = []

    start = time.time()

    for p in passes:
        print(f"  Running pass: {p.name} ...", end=" ")
        model = p.run(model)

        if verify_each_pass:
            try:
                verify(original, model, n_samples=n_verify_samples)
                print("✓")
            except AccuracyLossError as e:
                print(f"✗ FAILED — {e}")
                raise
        else:
            print("(verify skipped)")

        passes_applied.append(p.name)

    elapsed = time.time() - start
    nodes_after = count_nodes(model)
    size_after  = model_size_mb(model)

    onnx.save(model, output_path)

    report = {
        "nodes_before":   nodes_before,
        "nodes_after":    nodes_after,
        "nodes_delta_pct": round((nodes_before - nodes_after) / max(nodes_before, 1) * 100, 1),
        "size_before_mb": round(size_before, 2),
        "size_after_mb":  round(size_after, 2),
        "size_delta_pct": round((size_before - size_after) / max(size_before, 1e-9) * 100, 1),
        "passes_applied": passes_applied,
        "time_sec":       round(elapsed, 2),
    }

    _print_report(model_path, report)
    return report

def _print_report(model_path: str, r: dict):
    print(f"""
Model: {model_path}
─────────────────────────────────────────
Nodes before:      {r['nodes_before']}
Nodes after:       {r['nodes_after']} ({-r['nodes_delta_pct']:+.1f}%)
Size before:       {r['size_before_mb']} MB
Size after:        {r['size_after_mb']} MB ({-r['size_delta_pct']:+.1f}%)
Passes applied:    {', '.join(r['passes_applied']) if r['passes_applied'] else 'none'}
Time:              {r['time_sec']}s
""")


if __name__ == "__main__":
    import sys
    from passes import (
        EliminateDeadNodes,
        EliminateIdentityOps,
        EliminateUnusedInitializers,
        EliminateDuplicateConstants,
        EliminateRedundantTransposes,
        FoldConstants,
        SimplifyShapeChains,
        FuseConvBatchnorm,
        FuseConvRelu,
        FuseMatmulAdd,
        CleanupAttention,
    )

    if len(sys.argv) < 3:
        print("Usage: python optimizer.py input.onnx output.onnx")
        sys.exit(1)

    # M9: eleven passes now (CleanupAttention added)
    registered_passes = [
        EliminateDeadNodes(),
        EliminateIdentityOps(),
        EliminateUnusedInitializers(),
        EliminateDuplicateConstants(),
        EliminateRedundantTransposes(),
        FoldConstants(),
        SimplifyShapeChains(),
        FuseConvBatchnorm(),
        FuseConvRelu(),
        FuseMatmulAdd(),
        CleanupAttention(),
    ]

    optimize(
        model_path=sys.argv[1],
        output_path=sys.argv[2],
        passes=registered_passes,
    )
