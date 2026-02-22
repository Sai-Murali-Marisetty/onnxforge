"""
Run each pass individually and cumulatively on each model.
Records: nodes removed per pass, accuracy preserved.
This is the honest accounting of what the optimizer actually does.

Usage: python tests/experiments/exp_01_pass_attribution.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import onnx
from verify import verify


def all_passes():
    from passes.eliminate_dead_nodes import EliminateDeadNodes
    from passes.eliminate_identity_ops import EliminateIdentityOps
    from passes.eliminate_unused_initializers import EliminateUnusedInitializers
    from passes.eliminate_duplicate_constants import EliminateDuplicateConstants
    from passes.eliminate_redundant_transposes import EliminateRedundantTransposes
    from passes.fold_constants import FoldConstants
    from passes.simplify_shape_chains import SimplifyShapeChains
    from passes.fuse_conv_batchnorm import FuseConvBatchnorm
    from passes.fuse_conv_relu import FuseConvRelu
    from passes.fuse_matmul_add import FuseMatmulAdd
    from passes.cleanup_attention import CleanupAttention
    return [
        EliminateDeadNodes(), EliminateIdentityOps(),
        EliminateUnusedInitializers(), EliminateDuplicateConstants(),
        EliminateRedundantTransposes(), FoldConstants(),
        SimplifyShapeChains(), FuseConvBatchnorm(),
        FuseConvRelu(), FuseMatmulAdd(), CleanupAttention(),
    ]


def run_attribution(model_path, tolerance=1e-4):
    if not os.path.exists(model_path):
        print(f"  ⚠ Not found: {model_path}")
        return {}

    original = onnx.load(model_path)
    baseline_nodes = len(original.graph.node)
    baseline_size  = original.ByteSize() / 1024 / 1024

    print(f"\n{'='*65}")
    print(f"  {model_path}")
    print(f"  Baseline: {baseline_nodes} nodes  {baseline_size:.1f}MB")
    print(f"{'='*65}")
    print(f"  {'Pass':<38} {'isolated':>9} {'cumul':>7}  {'accuracy'}")
    print(f"  {'-'*38}  {'-'*9}  {'-'*7}  {'-'*12}")

    cumulative  = onnx.load(model_path)
    attribution = {}

    for p in all_passes():
        nodes_before_cumul = len(cumulative.graph.node)

        # Isolated run — just this one pass on the original
        try:
            m_isolated = p.run(onnx.load(model_path))
            isolated_delta = baseline_nodes - len(m_isolated.graph.node)
        except Exception as e:
            isolated_delta = f"ERR"

        # Cumulative run — this pass on top of all previous
        try:
            cumulative = p.run(cumulative)
            nodes_after_cumul = len(cumulative.graph.node)
            cumul_delta = nodes_before_cumul - nodes_after_cumul
        except Exception as e:
            cumul_delta = f"ERR:{str(e)[:20]}"
            nodes_after_cumul = nodes_before_cumul

        # Accuracy check on cumulative model
        try:
            rpt = verify(original, cumulative, n_samples=3, tolerance=tolerance)
            acc = f"✓ {rpt.max_diff:.1e}"
        except Exception as e:
            acc = f"✗ FAIL"

        iso_str   = f"{-isolated_delta:+d}" if isinstance(isolated_delta, int) else str(isolated_delta)
        cumul_str = f"{-cumul_delta:+d}"    if isinstance(cumul_delta, int) else str(cumul_delta)
        print(f"  {p.name:<38} {iso_str:>9}  {cumul_str:>7}  {acc}")
        attribution[p.name] = {"isolated": isolated_delta, "cumulative": cumul_delta}

    final = len(cumulative.graph.node)
    final_size = cumulative.ByteSize() / 1024 / 1024
    print(f"\n  TOTAL: {baseline_nodes} → {final} ({baseline_nodes - final:+d} nodes)")
    print(f"  SIZE:  {baseline_size:.1f}MB → {final_size:.1f}MB")
    return attribution


if __name__ == "__main__":
    models = [
        ("mobilenetv2-12.onnx",           1e-5),
        ("models/efficientnet-b0.onnx",   1e-4),
    ]
    all_results = {}
    for path, tol in models:
        all_results[path] = run_attribution(path, tol)

    # Summary table
    print(f"\n\n{'='*65}")
    print(f"  SUMMARY — Isolated node reduction per pass per model")
    print(f"{'='*65}")
    model_names = ["MobileNetV2", "EfficientNet"]
    print(f"  {'Pass':<38}", end="")
    for name in model_names:
        print(f" {name:>12}", end="")
    print()
    print(f"  {'-'*38}", end="")
    for _ in model_names:
        print(f" {'':>12}", end="")
    print()

    pass_names = [p.name for p in all_passes()]
    for pname in pass_names:
        print(f"  {pname:<38}", end="")
        for path, _ in models:
            res = all_results.get(path, {}).get(pname, {})
            iso = res.get("isolated", "?")
            val = f"{-iso:+d}" if isinstance(iso, int) else str(iso)
            print(f" {val:>12}", end="")
        print()
