#!/usr/bin/env python3
"""
Exp 23 — RoBERTa Anomaly Investigation

Why does RoBERTa get 11.8% latency improvement from only 5% node reduction,
while BERT (identical architecture) gets only 1.7% from the same 5%?

This is a 7x difference in efficiency. We need to understand why.
"""

import os
import time
import numpy as np
import onnx
import onnxruntime as ort
from collections import Counter

def profile_model(model_path, model_name, n_runs=50):
    """Profile a model with ORT profiling enabled."""
    print(f"\n{'='*70}")
    print(f"Profiling: {model_name}")
    print('='*70)
    
    # Load model
    model = onnx.load(model_path)
    graph = model.graph
    
    # Count ops
    op_counts = Counter(n.op_type for n in graph.node)
    print(f"Total nodes: {len(graph.node)}")
    print(f"Top ops: {op_counts.most_common(5)}")
    
    # Check input shapes
    print(f"\nInputs:")
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value > 0 else 'dyn' 
                 for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")
    
    # Enable profiling
    sess_opts = ort.SessionOptions()
    sess_opts.enable_profiling = True
    sess_opts.intra_op_num_threads = 1  # Single thread for consistent measurement
    
    session = ort.InferenceSession(model_path, sess_opts)
    
    # Create inputs
    inputs = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        # Use seq_len=128 for fair comparison
        if 'input_ids' in inp.name or 'attention_mask' in inp.name:
            shape = [1, 128]
        elif 'token_type_ids' in inp.name:
            shape = [1, 128]
        
        if inp.type == 'tensor(int64)':
            # token_type_ids should be 0 for RoBERTa
            if 'token_type_ids' in inp.name:
                inputs[inp.name] = np.zeros(shape, dtype=np.int64)
            else:
                inputs[inp.name] = np.ones(shape, dtype=np.int64)
        else:
            inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
    
    # Warmup
    for _ in range(5):
        session.run(None, inputs)
    
    # Profile
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, inputs)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    print(f"\nLatency: {avg_latency:.2f} ± {std_latency:.2f} ms")
    
    # Get profiling output
    prof_file = session.end_profiling()
    print(f"Profile saved to: {prof_file}")
    
    return {
        'model': model_name,
        'total_nodes': len(graph.node),
        'op_counts': op_counts,
        'avg_latency': avg_latency,
        'std_latency': std_latency,
        'prof_file': prof_file,
    }

def compare_weights(bert_path, roberta_path):
    """Compare weight shapes between BERT and RoBERTa."""
    print(f"\n{'='*70}")
    print("Weight Shape Comparison: BERT vs RoBERTa")
    print('='*70)
    
    bert = onnx.load(bert_path)
    roberta = onnx.load(roberta_path)
    
    def get_initializer_shapes(graph):
        shapes = {}
        for init in graph.initializer:
            shapes[init.name.split('/')[-1]] = list(init.dims)
        return shapes
    
    bert_shapes = get_initializer_shapes(bert.graph)
    roberta_shapes = get_initializer_shapes(roberta.graph)
    
    # Find differences
    print("\nNotable shape differences:")
    for name, bert_shape in bert_shapes.items():
        if name in roberta_shapes:
            roberta_shape = roberta_shapes[name]
            if bert_shape != roberta_shape:
                print(f"  {name[:40]}:")
                print(f"    BERT:    {bert_shape}")
                print(f"    RoBERTa: {roberta_shape}")
    
    # Check embedding sizes
    print("\nEmbedding analysis:")
    for name in ['word_embeddings.weight', 'embeddings.word_embeddings.weight']:
        if name in bert_shapes:
            print(f"  BERT {name}: {bert_shapes[name]}")
        if name in roberta_shapes:
            print(f"  RoBERTa {name}: {roberta_shapes[name]}")

def analyze_memory_access(model_path, model_name):
    """Estimate memory access patterns."""
    print(f"\n{'='*70}")
    print(f"Memory Access Analysis: {model_name}")
    print('='*70)
    
    model = onnx.load(model_path)
    graph = model.graph
    
    # Calculate total weight bytes
    total_weight_bytes = 0
    weight_shapes = []
    for init in graph.initializer:
        shape = list(init.dims)
        numel = 1
        for d in shape:
            numel *= d
        bytes_size = numel * 4  # float32
        total_weight_bytes += bytes_size
        if bytes_size > 1e6:  # > 1MB
            weight_shapes.append((init.name.split('/')[-1][:30], shape, bytes_size / 1e6))
    
    print(f"Total weight size: {total_weight_bytes / 1e6:.1f} MB")
    print(f"\nLargest weights (> 1MB):")
    for name, shape, mb in sorted(weight_shapes, key=lambda x: -x[2])[:10]:
        print(f"  {name}: {shape} ({mb:.1f} MB)")
    
    # Count Transpose nodes that touch weights
    transpose_count = sum(1 for n in graph.node if n.op_type == 'Transpose')
    matmul_count = sum(1 for n in graph.node if n.op_type == 'MatMul')
    
    print(f"\nOps that move memory:")
    print(f"  Transpose: {transpose_count}")
    print(f"  MatMul: {matmul_count}")
    
    return total_weight_bytes

def main():
    models_dir = "models"
    
    print("=" * 70)
    print("Exp 23: RoBERTa Anomaly Investigation")
    print("=" * 70)
    print("\nQuestion: Why does RoBERTa get 11.8% latency improvement from 5% node")
    print("reduction, while BERT gets only 1.7% from the same 5%?")
    
    # Profile original models
    bert_orig = profile_model(
        os.path.join(models_dir, "bert_base.onnx"), 
        "BERT (original)"
    )
    roberta_orig = profile_model(
        os.path.join(models_dir, "roberta_base.onnx"),
        "RoBERTa (original)"
    )
    
    # Profile optimized models
    bert_opt = profile_model(
        os.path.join(models_dir, "bert_base_opt.onnx"),
        "BERT (optimized)"
    )
    roberta_opt = profile_model(
        os.path.join(models_dir, "roberta_base_opt.onnx"),
        "RoBERTa (optimized)"
    )
    
    # Compare weights
    compare_weights(
        os.path.join(models_dir, "bert_base.onnx"),
        os.path.join(models_dir, "roberta_base.onnx")
    )
    
    # Memory analysis
    bert_mem = analyze_memory_access(
        os.path.join(models_dir, "bert_base.onnx"),
        "BERT"
    )
    roberta_mem = analyze_memory_access(
        os.path.join(models_dir, "roberta_base.onnx"),
        "RoBERTa"
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    bert_speedup = (bert_orig['avg_latency'] - bert_opt['avg_latency']) / bert_orig['avg_latency'] * 100
    roberta_speedup = (roberta_orig['avg_latency'] - roberta_opt['avg_latency']) / roberta_orig['avg_latency'] * 100
    
    print(f"\nLatency comparison:")
    print(f"  BERT:    {bert_orig['avg_latency']:.1f}ms → {bert_opt['avg_latency']:.1f}ms ({bert_speedup:+.1f}%)")
    print(f"  RoBERTa: {roberta_orig['avg_latency']:.1f}ms → {roberta_opt['avg_latency']:.1f}ms ({roberta_speedup:+.1f}%)")
    
    print(f"\nMemory size:")
    print(f"  BERT:    {bert_mem / 1e6:.1f} MB")
    print(f"  RoBERTa: {roberta_mem / 1e6:.1f} MB")
    
    print(f"\nEfficiency ratio:")
    print(f"  RoBERTa speedup / BERT speedup = {roberta_speedup / bert_speedup if bert_speedup > 0 else 'inf':.1f}x")
    
    print("\n" + "=" * 70)
    print("HYPOTHESIS")
    print("=" * 70)
    print("""
Possible explanations for the 7x efficiency difference:

1. VOCABULARY SIZE: RoBERTa has 50265 vocab vs BERT's 30522.
   This means larger embedding matrices that benefit more from
   Transpose elimination (more memory saved per removed Transpose).

2. MEASUREMENT VARIANCE: RoBERTa might have higher baseline variance,
   making the speedup measurement less reliable.

3. ORT CACHING: The removed Transpose nodes might have been causing
   cache misses that affected RoBERTa more due to larger weights.

4. CRITICAL PATH: The removed Transposes might be on the critical
   execution path in RoBERTa but parallel in BERT.

Next steps:
- Check variance (run 100+ seeds)
- Profile memory bandwidth usage
- Trace which specific nodes were on critical path
""")

if __name__ == "__main__":
    main()
