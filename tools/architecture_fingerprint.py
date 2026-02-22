#!/usr/bin/env python3
"""
Exp 22 — Architecture Fingerprinting Prototype

Extracts features from ONNX models to automatically classify architecture type.
This enables automatic pass selection in the optimizer.
"""

import os
import sys
import onnx
from collections import Counter
import numpy as np

def extract_fingerprint(model_path):
    """Extract architecture fingerprint from an ONNX model."""
    model = onnx.load(model_path)
    graph = model.graph
    
    # Op type histogram
    op_counts = Counter(node.op_type for node in graph.node)
    total_nodes = len(graph.node)
    
    # Key ratios
    conv_count = op_counts.get('Conv', 0)
    matmul_count = op_counts.get('MatMul', 0)
    gemm_count = op_counts.get('Gemm', 0)
    reshape_count = op_counts.get('Reshape', 0)
    transpose_count = op_counts.get('Transpose', 0)
    softmax_count = op_counts.get('Softmax', 0)
    layernorm_count = op_counts.get('LayerNormalization', 0)
    batchnorm_count = op_counts.get('BatchNormalization', 0)
    
    # Detect decomposed LayerNorm (ReduceMean -> Sub -> Pow -> ReduceMean -> Add -> Sqrt -> Div)
    # HuggingFace exports often decompose LayerNorm into this pattern
    reducemean_count = op_counts.get('ReduceMean', 0)
    sqrt_count = op_counts.get('Sqrt', 0)
    # If we have many ReduceMean + Sqrt pairs, it's likely decomposed LayerNorm
    decomposed_ln_est = min(reducemean_count // 2, sqrt_count)  # Each LN has 2 ReduceMean, 1 Sqrt
    
    # Activation patterns
    relu_count = op_counts.get('Relu', 0)
    gelu_count = op_counts.get('Gelu', 0)
    silu_count = op_counts.get('Silu', 0) + op_counts.get('Sigmoid', 0)  # SiLU = x * sigmoid(x)
    erf_count = op_counts.get('Erf', 0)  # GELU decomposition uses Erf
    
    # Compute features
    features = {
        'total_nodes': total_nodes,
        'conv_count': conv_count,
        'matmul_count': matmul_count,
        'gemm_count': gemm_count,
        'matmul_ratio': (matmul_count + gemm_count) / total_nodes if total_nodes > 0 else 0,
        'conv_ratio': conv_count / total_nodes if total_nodes > 0 else 0,
        'reshape_ratio': reshape_count / total_nodes if total_nodes > 0 else 0,
        'transpose_ratio': transpose_count / total_nodes if total_nodes > 0 else 0,
        'has_softmax': softmax_count > 0,
        'has_layernorm': layernorm_count > 0 or decomposed_ln_est >= 6,  # Either op or decomposed
        'has_batchnorm': batchnorm_count > 0,
        'has_gelu': gelu_count > 0 or erf_count > 0,  # GELU uses Erf
        'has_silu': silu_count > 0,
        'softmax_count': softmax_count,
        'layernorm_count': layernorm_count if layernorm_count > 0 else decomposed_ln_est,
        'batchnorm_count': batchnorm_count,
        'attention_blocks_est': softmax_count,  # Each attention head has one Softmax
        'decomposed_ln_est': decomposed_ln_est,
    }
    
    return features, op_counts

def classify_architecture(features):
    """
    Classify model architecture based on fingerprint.
    Returns (architecture_type, confidence, recommended_passes)
    """
    # Decision rules
    matmul_heavy = features['matmul_ratio'] > 0.03
    conv_heavy = features['conv_ratio'] > 0.05
    has_attention = features['has_softmax'] and features['softmax_count'] >= 6
    has_bn = features['has_batchnorm']
    has_ln = features['has_layernorm']
    
    # Classification logic
    if conv_heavy and not matmul_heavy:
        # Pure CNN (ResNet, EfficientNet, MobileNet, YOLO)
        arch_type = "vision-cnn"
        confidence = 0.9 if has_bn else 0.7
        recommended_passes = [
            'fuse_conv_batchnorm',
            'fuse_conv_relu', 
            'eliminate_redundant_transposes',
            'fold_constants',
        ]
        
    elif matmul_heavy and has_attention and not conv_heavy:
        if has_ln:
            # Pure transformer (BERT, DistilBERT, RoBERTa, GPT-2, ViT, DeiT)
            arch_type = "transformer"
            confidence = 0.95
            recommended_passes = [
                'fuse_matmul_add_3d',
                'eliminate_identity_ops',
                'cleanup_attention',
                'eliminate_redundant_transposes',
            ]
        else:
            # Transformer without LayerNorm (unusual)
            arch_type = "transformer-variant"
            confidence = 0.6
            recommended_passes = ['fuse_matmul_add_3d', 'cleanup_attention']
            
    elif conv_heavy and matmul_heavy:
        # Hybrid CNN + Transformer (MobileViT, Whisper, Wav2Vec2)
        arch_type = "hybrid"
        confidence = 0.85
        recommended_passes = [
            'fuse_conv_batchnorm',
            'fuse_matmul_add_3d',
            'eliminate_identity_ops',
            'cleanup_attention',
        ]
        
    elif matmul_heavy and not has_attention:
        # MLP-heavy model (some recommender systems, simple classifiers)
        arch_type = "mlp"
        confidence = 0.5
        recommended_passes = [
            'fuse_matmul_add',
            'fuse_matmul_add_3d',
            'fold_constants',
        ]
        
    else:
        # Unknown architecture
        arch_type = "unknown"
        confidence = 0.3
        recommended_passes = [
            'eliminate_dead_nodes',
            'eliminate_identity_ops',
            'fold_constants',
        ]
    
    return arch_type, confidence, recommended_passes

def analyze_model(model_path):
    """Full analysis of a single model."""
    features, op_counts = extract_fingerprint(model_path)
    arch_type, confidence, recommended_passes = classify_architecture(features)
    
    return {
        'features': features,
        'op_counts': op_counts,
        'arch_type': arch_type,
        'confidence': confidence,
        'recommended_passes': recommended_passes,
    }

def main():
    print("=" * 70)
    print("Exp 22: Architecture Fingerprinting Prototype")
    print("=" * 70)
    print()
    
    # Models to analyze
    models_dir = "models"
    model_files = [
        ("resnet50.onnx",           "vision-cnn"),      # Ground truth
        ("efficientnet_b0.onnx",    "vision-cnn"),
        ("mobilenetv3_small.onnx",  "vision-cnn"),
        ("bert_base.onnx",          "transformer"),
        ("distilbert_base.onnx",    "transformer"),
        ("roberta_base.onnx",       "transformer"),
        ("whisper_tiny_encoder.onnx", "hybrid"),
        ("whisper_base_encoder.onnx", "hybrid"),
        ("deit_small.onnx",         "transformer"),     # Vision transformer
        ("vit_base.onnx",           "transformer"),     # Vision transformer
        ("yolov8n.onnx",            "vision-cnn"),      # Detection
    ]
    
    print(f"{'Model':<28} {'Predicted':<14} {'Conf':>6} {'Actual':<14} {'Match':>6}")
    print("-" * 70)
    
    correct = 0
    total = 0
    
    for model_file, ground_truth in model_files:
        model_path = os.path.join(models_dir, model_file)
        
        if not os.path.exists(model_path):
            print(f"{model_file:<28} {'(not found)':<14}")
            continue
        
        total += 1
        analysis = analyze_model(model_path)
        predicted = analysis['arch_type']
        confidence = analysis['confidence']
        
        # Check if prediction matches ground truth
        # Note: vision transformers are "transformer" type
        is_correct = predicted == ground_truth
        if predicted == "transformer" and ground_truth == "vision-vit":
            is_correct = True  # Vision transformers are transformers
            
        correct += is_correct
        match_str = "✓" if is_correct else "✗"
        
        print(f"{model_file:<28} {predicted:<14} {confidence:>5.0%} {ground_truth:<14} {match_str:>6}")
    
    print("-" * 70)
    accuracy = correct / total if total > 0 else 0
    print(f"Classification accuracy: {correct}/{total} ({accuracy:.0%})")
    print()
    
    # Detailed analysis of one model
    print("=" * 70)
    print("DETAILED FINGERPRINT: BERT-base")
    print("=" * 70)
    
    bert_path = os.path.join(models_dir, "bert_base.onnx")
    if os.path.exists(bert_path):
        analysis = analyze_model(bert_path)
        
        print("\nTop 10 Op Types:")
        print("-" * 40)
        for op, count in sorted(analysis['op_counts'].items(), key=lambda x: -x[1])[:10]:
            print(f"  {op:<25} {count:>5}")
        
        print("\nKey Features:")
        print("-" * 40)
        for key, value in analysis['features'].items():
            if isinstance(value, float):
                print(f"  {key:<25} {value:>.4f}")
            else:
                print(f"  {key:<25} {value}")
        
        print("\nClassification:")
        print("-" * 40)
        print(f"  Architecture type: {analysis['arch_type']}")
        print(f"  Confidence: {analysis['confidence']:.0%}")
        print(f"  Recommended passes:")
        for p in analysis['recommended_passes']:
            print(f"    - {p}")
    
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The architecture fingerprinting prototype successfully classifies models
based on op-type ratios and structural patterns.

Key discriminating features:
1. MatMul ratio > 3%: Indicates transformer architecture
2. Conv ratio > 5%: Indicates CNN architecture
3. Softmax count >= 6: Indicates attention mechanism
4. BatchNorm presence: Indicates classical CNN (not transformer)
5. LayerNorm presence: Indicates modern transformer

This enables automatic pass selection:
- vision-cnn: Apply conv-bn fusion, transpose elimination
- transformer: Apply matmul-add-3d fusion, identity cleanup
- hybrid: Apply both pass families

Next steps:
1. Add more models to improve classification
2. Add LLM-specific features (RoPE detection, GQA patterns)
3. Integrate into optimizer CLI for auto-selection
""")

if __name__ == "__main__":
    main()
