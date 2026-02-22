"""
Experiment: Does pass order affect the final node count?
Shuffles pass order 5 times, compares to canonical order.
Expected: canonical is at least as good as shuffled; some orders are worse.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import random
import onnx


def get_passes():
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


def run_order(model_path, passes):
    model = onnx.load(model_path)
    for p in passes:
        try:
            model = p.run(model)
        except Exception:
            pass
    return len(model.graph.node)


if __name__ == "__main__":
    models = [
        "mobilenetv2-12.onnx",
        "models/efficientnet-b0.onnx",
    ]
    
    print("\nExperiment 07 — Pass Order Sensitivity\n")
    
    for path in models:
        if not os.path.exists(path):
            print(f"⚠ Not found: {path}")
            continue
            
        print(f"Model: {path}")
        print("-" * 50)
        
        canonical = run_order(path, get_passes())
        print(f"  Canonical order:  {canonical} nodes")
        
        shuffled_results = []
        for i in range(5):
            random.seed(i * 42)  # Reproducible shuffles
            p = get_passes()
            random.shuffle(p)
            count = run_order(path, p)
            shuffled_results.append(count)
            names = "→".join(x.name[:10] for x in p[:3]) + "→..."
            print(f"  Shuffle {i+1}:        {count} nodes  ({names})")
        
        best_shuffle = min(shuffled_results)
        worst_shuffle = max(shuffled_results)
        
        print(f"\n  Summary:")
        print(f"    Canonical:       {canonical} nodes")
        print(f"    Best shuffled:   {best_shuffle} nodes")
        print(f"    Worst shuffled:  {worst_shuffle} nodes")
        print(f"    Canonical best?  {'YES' if canonical <= best_shuffle else 'NO — shuffle found ' + str(best_shuffle)}")
        print(f"    Order sensitive? {'YES' if len(set(shuffled_results)) > 1 else 'NO (all same)'}")
        print()
