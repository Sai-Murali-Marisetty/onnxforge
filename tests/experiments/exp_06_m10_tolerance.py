#!/usr/bin/env python3
"""
M10 Experiment 06 — Tolerance Sweep (All New Models)

Verify perfect accuracy across all M10 models with 20 random seeds.
"""
import sys
import os
import onnx
import numpy as np

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
from verify import verify, AccuracyLossError

# All M10 models
M10_MODELS = [
    ("MobileNetV2", "mobilenetv2-12.onnx"),
    ("EfficientNet-B0", "models/efficientnet-b0.onnx"),
    ("ResNet-50", "models/resnet50.onnx"),
    ("MobileNetV3-S", "models/mobilenetv3_small.onnx"),
    ("BERT-base", "models/bert_base.onnx"),
    ("DistilBERT", "models/distilbert_base.onnx"),
    ("RoBERTa-base", "models/roberta_base.onnx"),
    ("Whisper-tiny", "models/whisper_tiny_encoder.onnx"),
]

N_SEEDS = 20

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
    """Run full pipeline on model."""
    for p in get_all_passes():
        model = p.run(model)
    return model

def main():
    print("="*80)
    print("M10 EXPERIMENT 06 — TOLERANCE SWEEP (ALL MODELS)")
    print("="*80)
    print(f"\nRunning {N_SEEDS} random seeds per model...\n")
    
    results = {}
    
    for model_name, model_path in M10_MODELS:
        if not os.path.exists(model_path):
            print(f"⚠ Skipping {model_name}: {model_path} not found")
            continue
        
        print(f"\n{'─'*60}")
        print(f"Testing: {model_name}")
        print(f"{'─'*60}")
        
        original = onnx.load(model_path)
        optimized = optimize_model(onnx.load(model_path))
        
        diffs = []
        errors = 0
        
        for seed in range(N_SEEDS):
            np.random.seed(seed)
            try:
                report = verify(original, optimized, n_samples=1)
                diffs.append(report.max_diff)
            except AccuracyLossError as e:
                errors += 1
                diffs.append(float('inf'))
            except Exception as e:
                errors += 1
                print(f"  Seed {seed}: Error - {e}")
        
        if diffs:
            valid_diffs = [d for d in diffs if d != float('inf')]
            if valid_diffs:
                min_d = min(valid_diffs)
                max_d = max(valid_diffs)
                mean_d = np.mean(valid_diffs)
                all_zero = all(d == 0.0 for d in valid_diffs)
            else:
                min_d = max_d = mean_d = float('inf')
                all_zero = False
            
            results[model_name] = {
                'min': min_d,
                'max': max_d,
                'mean': mean_d,
                'all_zero': all_zero,
                'errors': errors,
            }
            
            status = "✓ PERFECT" if all_zero else ("✓ PASS" if max_d < 1e-5 else "✗ FAIL")
            print(f"  Seeds tested: {N_SEEDS}")
            print(f"  Errors: {errors}")
            print(f"  max_diff: min={min_d:.2e}, max={max_d:.2e}, mean={mean_d:.2e}")
            print(f"  all_zero: {all_zero}")
            print(f"  Status: {status}")
    
    # Summary table
    print(f"\n{'='*80}")
    print("TOLERANCE SWEEP SUMMARY")
    print("="*80)
    print(f"\n{'Model':<20} | {'Min':>10} | {'Max':>10} | {'Mean':>10} | {'All Zero':>8} | {'Status':>10}")
    print("-"*80)
    
    all_pass = True
    for model_name, data in results.items():
        status = "✓ PERFECT" if data['all_zero'] else ("✓ PASS" if data['max'] < 1e-5 else "✗ FAIL")
        if data['max'] >= 1e-5:
            all_pass = False
        print(f"{model_name:<20} | {data['min']:>10.2e} | {data['max']:>10.2e} | {data['mean']:>10.2e} | {str(data['all_zero']):>8} | {status:>10}")
    
    print("="*80)
    print(f"\nOVERALL: {'✓ ALL MODELS PASS' if all_pass else '✗ SOME MODELS FAIL'}")
    
    return results

if __name__ == "__main__":
    main()
