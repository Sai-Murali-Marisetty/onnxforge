#!/usr/bin/env python3
"""
M10 Experiment 01 — Full Pass Attribution Matrix

Run each pass in isolation on every model. Record the exact node count change.
This is THE master table - every row is a model, every column is a pass.
"""
import sys
import os
import onnx
import copy

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

def get_all_passes():
    return [
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
    ]

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

def count_nodes(model):
    return len(model.graph.node)

def run_single_pass(model, pass_class):
    """Run a single pass and return node delta."""
    model_copy = copy.deepcopy(model)
    before = count_nodes(model_copy)
    
    try:
        pass_instance = pass_class()
        model_copy = pass_instance.run(model_copy)
        after = count_nodes(model_copy)
        return before - after  # Positive = nodes removed
    except Exception as e:
        return f"ERR"

def main():
    print("="*120)
    print("M10 EXPERIMENT 01 — FULL PASS ATTRIBUTION MATRIX")
    print("="*120)
    
    passes = get_all_passes()
    pass_names = [p.__class__.__name__ for p in [cls() for cls in passes]]
    
    # Short names for display
    short_names = {
        'EliminateDeadNodes': 'dead_nodes',
        'EliminateIdentityOps': 'identity_ops', 
        'EliminateUnusedInitializers': 'unused_init',
        'EliminateDuplicateConstants': 'dup_const',
        'EliminateRedundantTransposes': 'transp',
        'FoldConstants': 'fold_const',
        'SimplifyShapeChains': 'shape_chains',
        'FuseConvBatchnorm': 'conv_bn',
        'FuseConvRelu': 'conv_relu',
        'FuseMatmulAdd': 'matmul_add',
        'CleanupAttention': 'attention',
    }
    
    # Results matrix
    results = {}
    
    for model_name, model_path in M10_MODELS:
        if not os.path.exists(model_path):
            print(f"\n⚠ Skipping {model_name}: {model_path} not found")
            continue
            
        print(f"\n{'─'*60}")
        print(f"Testing: {model_name}")
        print(f"{'─'*60}")
        
        model = onnx.load(model_path)
        before_total = count_nodes(model)
        
        results[model_name] = {'before': before_total, 'passes': {}}
        
        for pass_class in passes:
            pass_name = pass_class.__name__
            delta = run_single_pass(model, pass_class)
            results[model_name]['passes'][pass_name] = delta
            short = short_names.get(pass_name, pass_name[:10])
            print(f"  {short:<15} Δ = {delta:>5}")
        
        # Run full pipeline
        model_copy = copy.deepcopy(model)
        for pass_class in passes:
            try:
                pass_instance = pass_class()
                model_copy = pass_instance.run(model_copy)
            except:
                pass
        after_total = count_nodes(model_copy)
        results[model_name]['after'] = after_total
        results[model_name]['total_delta'] = before_total - after_total
        
        pct = 100 * (before_total - after_total) / before_total if before_total > 0 else 0
        print(f"  {'─'*40}")
        print(f"  TOTAL: {before_total} → {after_total} ({pct:.1f}% reduction)")
    
    # Print summary matrix
    print("\n" + "="*140)
    print("PASS ATTRIBUTION MATRIX (Δ nodes, isolated runs)")
    print("="*140)
    
    # Header
    cols = list(short_names.values())
    header = f"{'Model':<18} |"
    for col in cols:
        header += f" {col[:8]:>8} |"
    header += f" {'TOTAL':>8}"
    print(header)
    print("-"*140)
    
    # Data rows
    for model_name, data in results.items():
        row = f"{model_name:<18} |"
        for pass_name in short_names.keys():
            val = data['passes'].get(pass_name, '-')
            if isinstance(val, int):
                row += f" {val:>8} |"
            else:
                row += f" {str(val):>8} |"
        row += f" {data.get('total_delta', 0):>8}"
        print(row)
    
    print("="*140)
    
    # Key findings
    print("\nKEY FINDINGS:")
    print("─"*60)
    
    # Check hard gates
    gates = {
        'transpose_fires': False,
        'fold_const_fires': False,
        'matmul_add_fires': False,
        'attention_fires': False,
    }
    
    for model_name, data in results.items():
        transp = data['passes'].get('EliminateRedundantTransposes', 0)
        fold = data['passes'].get('FoldConstants', 0)
        matmul = data['passes'].get('FuseMatmulAdd', 0)
        attn = data['passes'].get('CleanupAttention', 0)
        
        if isinstance(transp, int) and transp > 0:
            gates['transpose_fires'] = True
            print(f"✓ eliminate_redundant_transposes: -{transp} on {model_name}")
        if isinstance(fold, int) and fold > 0:
            gates['fold_const_fires'] = True
            print(f"✓ fold_constants: -{fold} on {model_name}")
        if isinstance(matmul, int) and matmul > 0:
            gates['matmul_add_fires'] = True
            print(f"✓ fuse_matmul_add: -{matmul} on {model_name}")
        if isinstance(attn, int) and attn > 0:
            gates['attention_fires'] = True
            print(f"✓ cleanup_attention: -{attn} on {model_name}")
    
    # Conv+BN check
    for model_name, data in results.items():
        bn = data['passes'].get('FuseConvBatchnorm', 0)
        if isinstance(bn, int) and bn > 0:
            print(f"✓ fuse_conv_batchnorm: -{bn} on {model_name}")
    
    print("\n" + "─"*60)
    print("HARD GATE STATUS:")
    print(f"  Transpose fires on real model: {'✓' if gates['transpose_fires'] else '✗'}")
    print(f"  Fold constants on Transformer: {'✓' if gates['fold_const_fires'] else '✗'}")
    print(f"  MatMul+Add fires on Transformer: {'✓' if gates['matmul_add_fires'] else '✗'}")
    print(f"  Attention cleanup on BERT: {'✓' if gates['attention_fires'] else '✗'}")
    
    return results

if __name__ == "__main__":
    results = main()
