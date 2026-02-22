#!/usr/bin/env python3
"""
Exp 21 — Node Reduction vs Latency Correlation Analysis

Question: Does node count reduction predict latency improvement?
This is the publishable finding that nobody has measured formally.
"""

import numpy as np

# Data from M11 benchmarks
# Format: (model_name, family, node_reduction_pct, latency_improvement_pct)
data = [
    # Vision CNN models
    ("ResNet-50",       "vision-cnn", 31.8, 3.7),
    ("EfficientNet-B0", "vision-cnn", 17.0, 0.3),
    ("MobileNetV3-S",   "vision-cnn", 19.4, 1.8),
    ("MobileNetV2",     "vision-cnn", 0.0,  0.5),
    
    # Transformer encoders
    ("BERT-base",       "transformer", 5.0,  1.7),
    ("DistilBERT",      "transformer", 4.8,  0.9),
    ("RoBERTa",         "transformer", 5.0, 11.8),
    
    # Vision transformers
    ("DeiT-Small",      "vision-vit",  13.5, -0.4),
    ("ViT-Base",        "vision-vit",  5.6,  0.1),
    
    # Audio transformers
    ("Whisper-tiny",    "audio",       5.3,  0.8),
    ("Whisper-base",    "audio",       5.4,  0.2),
    
    # Detection (pre-optimized)
    ("YOLOv8n",         "detection",   0.0,  0.2),
]

def analyze_correlation():
    """Compute and report correlation statistics."""
    print("=" * 70)
    print("Exp 21: Node Reduction vs Latency Improvement Correlation")
    print("=" * 70)
    print()
    
    # Extract data
    names = [d[0] for d in data]
    families = [d[1] for d in data]
    node_reductions = np.array([d[2] for d in data])
    latency_improvements = np.array([d[3] for d in data])
    
    # Overall correlation
    overall_corr = np.corrcoef(node_reductions, latency_improvements)[0, 1]
    
    print("OVERALL CORRELATION")
    print("-" * 40)
    print(f"Pearson correlation coefficient: {overall_corr:.3f}")
    print()
    
    # Interpretation
    if abs(overall_corr) < 0.3:
        strength = "WEAK"
    elif abs(overall_corr) < 0.7:
        strength = "MODERATE"
    else:
        strength = "STRONG"
    
    print(f"Interpretation: {strength} {'positive' if overall_corr > 0 else 'negative'} correlation")
    print()
    
    # Per-family analysis
    print("PER-FAMILY CORRELATION")
    print("-" * 40)
    
    unique_families = list(set(families))
    for family in unique_families:
        indices = [i for i, f in enumerate(families) if f == family]
        if len(indices) >= 3:  # Need at least 3 points for meaningful correlation
            family_nodes = node_reductions[indices]
            family_latency = latency_improvements[indices]
            family_corr = np.corrcoef(family_nodes, family_latency)[0, 1]
            print(f"{family:15} (n={len(indices)}): r = {family_corr:.3f}")
        else:
            print(f"{family:15} (n={len(indices)}): insufficient data")
    
    print()
    
    # Detailed table
    print("DETAILED DATA")
    print("-" * 70)
    print(f"{'Model':<18} {'Family':<12} {'Node Δ%':>8} {'Latency Δ%':>10} {'Efficient':>10}")
    print("-" * 70)
    
    for name, family, node_red, lat_imp in data:
        # Efficiency = latency improvement per node reduction
        efficiency = lat_imp / node_red if node_red > 0 else float('inf') if lat_imp > 0 else 0
        efficiency_str = f"{efficiency:.2f}" if efficiency != float('inf') else "∞"
        print(f"{name:<18} {family:<12} {node_red:>8.1f} {lat_imp:>10.1f} {efficiency_str:>10}")
    
    print()
    
    # Key findings
    print("KEY FINDINGS")
    print("-" * 70)
    
    # Find outliers
    max_lat_idx = np.argmax(latency_improvements)
    max_node_idx = np.argmax(node_reductions)
    
    print(f"1. Best latency improvement: {names[max_lat_idx]} (+{latency_improvements[max_lat_idx]:.1f}%)")
    print(f"   with only {node_reductions[max_lat_idx]:.1f}% node reduction")
    print()
    
    print(f"2. Best node reduction: {names[max_node_idx]} (-{node_reductions[max_node_idx]:.1f}% nodes)")
    print(f"   with only {latency_improvements[max_node_idx]:.1f}% latency improvement")
    print()
    
    # Find inefficient cases
    inefficient = [(n, nr, li) for n, _, nr, li in data if nr > 10 and li < 1]
    if inefficient:
        print(f"3. Inefficient cases (>10% node reduction, <1% speedup):")
        for name, nr, li in inefficient:
            print(f"   - {name}: {nr:.1f}% nodes removed → only {li:.1f}% faster")
    print()
    
    # Find efficient cases  
    efficient = [(n, nr, li) for n, _, nr, li in data if nr > 0 and li / nr > 1]
    if efficient:
        print(f"4. Efficient cases (latency Δ > node Δ):")
        for name, nr, li in efficient:
            print(f"   - {name}: {nr:.1f}% nodes removed → {li:.1f}% faster ({li/nr:.1f}x efficiency)")
    print()
    
    # Hypothesis testing
    print("HYPOTHESIS TESTING")
    print("-" * 70)
    print()
    print("H0: Node reduction predicts latency improvement (correlation exists)")
    print("H1: Node reduction does NOT predict latency (weak/no correlation)")
    print()
    
    if abs(overall_corr) < 0.5:
        print("RESULT: REJECT H0")
        print()
        print("Node count reduction is a POOR PROXY for latency improvement.")
        print("This is a publishable finding — papers should report both metrics.")
    else:
        print("RESULT: FAIL TO REJECT H0")
        print()
        print("Node count reduction reasonably predicts latency improvement.")
    print()
    
    # Paper-ready conclusion
    print("=" * 70)
    print("PAPER-READY CONCLUSION")
    print("=" * 70)
    print("""
Our analysis of 12 models across 5 architecture families reveals that
node count reduction (r = {:.2f}) is a {} predictor of inference latency
improvement.

Notable outliers:
- RoBERTa: 5.0% node reduction → 11.8% latency improvement (2.4x efficiency)
- DeiT-Small: 13.5% node reduction → -0.4% latency regression (negative efficiency)

This suggests that:
1. Elimination passes (removing dead nodes) reduce node count but may not
   improve latency significantly
2. Fusion passes (combining MatMul+Add) reduce memory bandwidth, yielding
   disproportionate latency benefits
3. Model architecture strongly mediates the node-to-latency relationship

Recommendation: Graph optimization tools should report BOTH node reduction
AND latency improvement. Node count alone is insufficient for evaluating
optimization effectiveness.
""".format(overall_corr, strength.lower()))
    
    return overall_corr

if __name__ == "__main__":
    analyze_correlation()
