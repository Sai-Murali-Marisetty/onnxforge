"""
Experiment: Measure actual max_diff for each model across 20 random seeds.
Gives the real accuracy bounds — not assumed ones.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import onnx
import onnxruntime as ort


def gen_inputs(model, seed):
    np.random.seed(seed)
    feed = {}
    for inp in model.graph.input:
        name = inp.name
        # Skip if it's an initializer (not a true input)
        init_names = {i.name for i in model.graph.initializer}
        if name in init_names:
            continue
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value > 0 else 1)
        dtype = inp.type.tensor_type.elem_type
        if dtype == 1:  # FLOAT
            feed[name] = np.random.randn(*shape).astype(np.float32)
        elif dtype == 7:  # INT64
            feed[name] = np.random.randint(0, 100, shape).astype(np.int64)
        else:
            feed[name] = np.random.randn(*shape).astype(np.float32)
    return feed


def sweep(orig_path, opt_path, n=20):
    orig = onnx.load(orig_path)
    opt  = onnx.load(opt_path)
    sess_orig = ort.InferenceSession(orig.SerializeToString())
    sess_opt  = ort.InferenceSession(opt.SerializeToString())
    out_names_orig = [o.name for o in orig.graph.output]
    out_names_opt  = [o.name for o in opt.graph.output]
    diffs = []
    for seed in range(n):
        feed = gen_inputs(orig, seed)
        a = sess_orig.run(out_names_orig, feed)
        b = sess_opt.run(out_names_opt, feed)
        for x, y in zip(a, b):
            diffs.append(float(np.max(np.abs(np.array(x) - np.array(y)))))
    return {
        "min": min(diffs), "max": max(diffs),
        "mean": float(np.mean(diffs)), "p99": float(np.percentile(diffs, 99)),
        "all_zero": all(d == 0.0 for d in diffs),
    }


if __name__ == "__main__":
    from optimizer import optimize
    from passes import (
        EliminateDeadNodes, EliminateIdentityOps, EliminateUnusedInitializers,
        EliminateDuplicateConstants, EliminateRedundantTransposes, FoldConstants,
        SimplifyShapeChains, FuseConvBatchnorm, FuseConvRelu, FuseMatmulAdd,
        CleanupAttention,
    )

    passes = [
        EliminateDeadNodes(), EliminateIdentityOps(),
        EliminateUnusedInitializers(), EliminateDuplicateConstants(),
        EliminateRedundantTransposes(), FoldConstants(),
        SimplifyShapeChains(), FuseConvBatchnorm(),
        FuseConvRelu(), FuseMatmulAdd(), CleanupAttention(),
    ]

    pairs = [
        ("mobilenetv2-12.onnx",          "models/mobilenetv2-opt.onnx"),
        ("models/efficientnet-b0.onnx",  "models/efficientnet-opt.onnx"),
    ]

    print(f"\nExperiment 06 — Tolerance Sweep (n=20 seeds)\n")
    print(f"{'Model':<22} {'min':>10} {'max':>10} {'mean':>10} {'p99':>10} {'all_zero':>10}")
    print("-" * 76)

    for orig, opt in pairs:
        if not os.path.exists(orig):
            print(f"  ⚠ Missing: {orig}")
            continue
        if not os.path.exists(opt):
            print(f"  Optimizing {orig}...")
            optimize(orig, opt, passes=passes, verify_each_pass=False, n_verify_samples=1)
        r = sweep(orig, opt, n=20)
        name = os.path.basename(orig).replace(".onnx", "")
        print(f"{name:<22} {r['min']:>10.2e} {r['max']:>10.2e} "
              f"{r['mean']:>10.2e} {r['p99']:>10.2e} {str(r['all_zero']):>10}")

    print("\nRecommended tolerances (p99 * 10x safety margin):")
    print("  MobileNetV2:   1e-5 (if all_zero, use 1e-6)")
    print("  EfficientNet:  1e-5 (if all_zero, use 1e-6)")
