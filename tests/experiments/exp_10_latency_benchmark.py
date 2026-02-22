#!/usr/bin/env python3
"""
M10 Experiment 10 — Latency Benchmark

First real performance numbers on ORT CPU.
Measure median latency before/after optimization.
"""
import sys
import os
import time
import onnx
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

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

# Models to benchmark
MODELS = [
    ("EfficientNet-B0", "models/efficientnet-b0.onnx", {"input": (1, 3, 224, 224)}),
    ("ResNet-50", "models/resnet50.onnx", {"input": (1, 3, 224, 224)}),
    ("MobileNetV3-S", "models/mobilenetv3_small.onnx", {"input": (1, 3, 224, 224)}),
    ("DistilBERT", "models/distilbert_base.onnx", {
        "input_ids": (1, 128),
        "attention_mask": (1, 128),
    }),
    ("Whisper-tiny", "models/whisper_tiny_encoder.onnx", {"input_features": (1, 80, 3000)}),
]

N_WARMUP = 5
N_RUNS = 50

def get_all_passes():
    return [
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

def optimize_model(model):
    for p in get_all_passes():
        model = p.run(model)
    return model

def generate_input(input_specs):
    """Generate random input matching the model's expected shapes."""
    feed = {}
    for name, shape in input_specs.items():
        if 'ids' in name.lower() or 'mask' in name.lower():
            feed[name] = np.random.randint(0, 100, shape).astype(np.int64)
        else:
            feed[name] = np.random.randn(*shape).astype(np.float32)
    return feed

def benchmark_model(model_bytes, input_specs, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Run inference and return median latency in ms."""
    sess = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
    output_names = [o.name for o in sess.get_outputs()]
    
    feed = generate_input(input_specs)
    
    # Warmup
    for _ in range(n_warmup):
        sess.run(output_names, feed)
    
    # Benchmark
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        sess.run(output_names, feed)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
    
    return np.median(latencies)

def main():
    print("="*80)
    print("M10 EXPERIMENT 10 — LATENCY BENCHMARK")
    print("="*80)
    print(f"\nSettings: {N_WARMUP} warmup, {N_RUNS} runs, median latency")
    print(f"Runtime: ONNX Runtime {ort.__version__} (CPU)")
    
    results = []
    
    for model_name, model_path, input_specs in MODELS:
        if not os.path.exists(model_path):
            print(f"\n⚠ Skipping {model_name}: {model_path} not found")
            continue
        
        print(f"\n{'─'*60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'─'*60}")
        
        # Load and optimize
        original = onnx.load(model_path)
        optimized = optimize_model(onnx.load(model_path))
        
        nodes_before = len(original.graph.node)
        nodes_after = len(optimized.graph.node)
        
        print(f"  Nodes: {nodes_before} → {nodes_after} ({nodes_before-nodes_after} removed)")
        
        # Benchmark original
        print(f"  Running {N_RUNS} inferences (original)...", end=" ", flush=True)
        lat_before = benchmark_model(original.SerializeToString(), input_specs)
        print(f"{lat_before:.2f} ms")
        
        # Benchmark optimized
        print(f"  Running {N_RUNS} inferences (optimized)...", end=" ", flush=True)
        lat_after = benchmark_model(optimized.SerializeToString(), input_specs)
        print(f"{lat_after:.2f} ms")
        
        speedup = lat_before / lat_after if lat_after > 0 else 1.0
        improvement = (lat_before - lat_after) / lat_before * 100 if lat_before > 0 else 0
        
        results.append({
            'name': model_name,
            'nodes_before': nodes_before,
            'nodes_after': nodes_after,
            'lat_before': lat_before,
            'lat_after': lat_after,
            'speedup': speedup,
            'improvement': improvement,
        })
        
        print(f"  Speedup: {speedup:.3f}x ({improvement:+.1f}%)")
    
    # Summary table
    print(f"\n{'='*90}")
    print("LATENCY BENCHMARK SUMMARY")
    print("="*90)
    print(f"\n{'Model':<18} | {'Before (ms)':>12} | {'After (ms)':>12} | {'Speedup':>8} | {'Node Δ':>8} | {'Improve':>8}")
    print("-"*90)
    
    for r in results:
        delta = r['nodes_before'] - r['nodes_after']
        print(f"{r['name']:<18} | {r['lat_before']:>12.2f} | {r['lat_after']:>12.2f} | {r['speedup']:>7.3f}x | {delta:>8} | {r['improvement']:>+7.1f}%")
    
    print("="*90)
    
    # Check hard gate
    any_improvement = any(r['improvement'] > 2.0 for r in results)
    print(f"\nHARD GATE: Latency improvement >2% on at least one model: {'✓ PASS' if any_improvement else '✗ FAIL'}")
    
    return results

if __name__ == "__main__":
    main()
