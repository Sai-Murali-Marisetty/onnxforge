#!/usr/bin/env python3
"""
Post-Optimization Forensics

Run forensics AFTER the pipeline to understand what patterns remain.
Key question: Why did only 72 of 84 MatMul+Add pairs fold in BERT?
"""

import os
import sys
import onnx
import numpy as np
from collections import defaultdict

def get_initializer_names(graph):
    """Get set of all initializer names."""
    return {init.name for init in graph.initializer}

def find_matmul_add_pairs(graph, initializer_names):
    """Find all MatMul+Add patterns, classify why they did/didn't fold."""
    
    # Build consumer map
    consumers = defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            consumers[inp].append(node)
    
    # Build producer map
    producers = {}
    for node in graph.node:
        for out in node.output:
            producers[out] = node
    
    pairs = []
    
    for node in graph.node:
        if node.op_type != 'MatMul':
            continue
        
        # Check if MatMul output goes to exactly one Add
        matmul_out = node.output[0]
        matmul_consumers = consumers.get(matmul_out, [])
        
        if len(matmul_consumers) != 1:
            continue
        
        add_node = matmul_consumers[0]
        if add_node.op_type != 'Add':
            continue
        
        # Found a MatMul -> Add pair. Now classify it.
        matmul_input_a = node.input[0]
        matmul_input_b = node.input[1]
        
        # Check input B (weight) source
        weight_source = "unknown"
        weight_is_transposed = False
        weight_from_initializer = False
        
        if matmul_input_b in initializer_names:
            weight_source = "direct_initializer"
            weight_from_initializer = True
        elif matmul_input_b in producers:
            producer = producers[matmul_input_b]
            if producer.op_type == 'Transpose':
                transpose_input = producer.input[0]
                if transpose_input in initializer_names:
                    weight_source = "transpose_of_initializer"
                    weight_from_initializer = True
                    weight_is_transposed = True
                    
                    # Check perm
                    perm = None
                    for attr in producer.attribute:
                        if attr.name == 'perm':
                            perm = list(attr.ints)
                    
                    if perm == [1, 0]:
                        weight_source = "transpose_[1,0]_of_initializer"
                    else:
                        weight_source = f"transpose_{perm}_of_initializer"
                else:
                    weight_source = f"transpose_of_{producer.input[0][:30]}"
            else:
                weight_source = f"{producer.op_type}_output"
        
        # Check bias source
        add_other_input = add_node.input[0] if add_node.input[1] == matmul_out else add_node.input[1]
        bias_source = "unknown"
        
        if add_other_input in initializer_names:
            bias_source = "direct_initializer"
        elif add_other_input in producers:
            bias_prod = producers[add_other_input]
            bias_source = f"{bias_prod.op_type}_output"
        
        # Determine fold status
        can_fold = False
        fold_blocker = "unknown"
        
        if weight_source == "transpose_[1,0]_of_initializer" and bias_source == "direct_initializer":
            can_fold = True
            fold_blocker = "should_have_folded"
        elif weight_source == "direct_initializer" and bias_source == "direct_initializer":
            can_fold = False
            fold_blocker = "weight_already_transposed"
        elif "transpose" in weight_source and bias_source == "direct_initializer":
            can_fold = False
            fold_blocker = f"non_standard_transpose_perm"
        elif bias_source != "direct_initializer":
            can_fold = False
            fold_blocker = f"bias_not_initializer ({bias_source})"
        else:
            fold_blocker = f"weight_source={weight_source}"
        
        pairs.append({
            'matmul_name': node.name,
            'add_name': add_node.name,
            'weight_source': weight_source,
            'bias_source': bias_source,
            'can_fold': can_fold,
            'fold_blocker': fold_blocker,
        })
    
    return pairs

def analyze_model(model_path, label=""):
    """Full analysis of MatMul+Add patterns in a model."""
    print(f"\n{'='*70}")
    print(f"Post-Optimization Forensics: {os.path.basename(model_path)} {label}")
    print('='*70)
    
    model = onnx.load(model_path)
    graph = model.graph
    init_names = get_initializer_names(graph)
    
    pairs = find_matmul_add_pairs(graph, init_names)
    
    print(f"\nTotal MatMul -> Add pairs found: {len(pairs)}")
    
    # Classify by fold status
    by_blocker = defaultdict(list)
    for p in pairs:
        by_blocker[p['fold_blocker']].append(p)
    
    print(f"\nBreakdown by fold status:")
    print("-" * 50)
    for blocker, items in sorted(by_blocker.items(), key=lambda x: -len(x[1])):
        print(f"  {blocker}: {len(items)}")
        if len(items) <= 5:
            for item in items:
                print(f"    - MatMul: {item['matmul_name'][:40]}")
    
    return pairs

def compare_before_after(before_path, after_path, model_name):
    """Compare MatMul+Add patterns before and after optimization."""
    print(f"\n{'#'*70}")
    print(f"# BEFORE vs AFTER: {model_name}")
    print('#'*70)
    
    before_pairs = analyze_model(before_path, "(BEFORE)")
    after_pairs = analyze_model(after_path, "(AFTER)")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"Before optimization: {len(before_pairs)} MatMul+Add pairs")
    print(f"After optimization:  {len(after_pairs)} MatMul+Add pairs")
    print(f"Pairs folded:        {len(before_pairs) - len(after_pairs)}")
    
    if len(after_pairs) > 0:
        print(f"\n⚠️  {len(after_pairs)} pairs remain unfolded!")
        print("\nRemaining pairs analysis:")
        for p in after_pairs[:10]:
            print(f"  - {p['fold_blocker']}: {p['matmul_name'][:50]}")

def main():
    models_dir = "models"
    
    # BERT before/after
    bert_before = os.path.join(models_dir, "bert_base.onnx")
    bert_after = os.path.join(models_dir, "bert_base_opt.onnx")
    
    if os.path.exists(bert_before):
        analyze_model(bert_before, "(ORIGINAL)")
    
    if os.path.exists(bert_after):
        analyze_model(bert_after, "(OPTIMIZED)")
    
    if os.path.exists(bert_before) and os.path.exists(bert_after):
        compare_before_after(bert_before, bert_after, "BERT-base")
    
    # Also check DistilBERT
    distilbert_before = os.path.join(models_dir, "distilbert_base.onnx")
    distilbert_after = os.path.join(models_dir, "distilbert_base_opt.onnx")
    
    if os.path.exists(distilbert_before) and os.path.exists(distilbert_after):
        compare_before_after(distilbert_before, distilbert_after, "DistilBERT")

if __name__ == "__main__":
    main()
