"""
Deep audit of a transformer ONNX graph.
Produces a structured report of every optimization opportunity.

Usage:
    python -m tools.graph_forensics models/bert_base.onnx
    python -m tools.graph_forensics models/*.onnx --summary
"""

import onnx
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
import numpy as np


def build_node_output_map(graph) -> Dict[str, onnx.NodeProto]:
    """Map output name -> producing node."""
    return {out: node for node in graph.node for out in node.output}


def build_node_input_map(graph) -> Dict[str, List[onnx.NodeProto]]:
    """Map output name -> consuming nodes."""
    result = defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            result[inp].append(node)
    return result


def get_initializer_names(graph) -> Set[str]:
    """Get all initializer names."""
    return {init.name for init in graph.initializer}


def get_value_info_shapes(graph) -> Dict[str, List[int]]:
    """Build map of name -> shape from value_info and graph inputs."""
    shapes = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.type.HasField('tensor_type'):
            shape = []
            for dim in vi.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)  # dynamic
            shapes[vi.name] = shape
    return shapes


def find_3d_matmuls(graph) -> List[Dict]:
    """Find MatMul operations with 3D+ inputs (batch dimension)."""
    shapes = get_value_info_shapes(graph)
    inits = get_initializer_names(graph)
    
    results = []
    for node in graph.node:
        if node.op_type == "MatMul":
            # Check input shapes
            for inp in node.input:
                if inp in shapes:
                    shape = shapes[inp]
                    if len(shape) >= 3:
                        results.append({
                            "node_name": node.name,
                            "input": inp,
                            "shape": shape,
                            "rank": len(shape)
                        })
                        break
    return results


def find_3d_matmul_add_pairs(graph) -> List[Dict]:
    """Find MatMul(3D) → Add patterns (the known M10 gap)."""
    output_map = build_node_output_map(graph)
    input_map = build_node_input_map(graph)
    shapes = get_value_info_shapes(graph)
    inits = get_initializer_names(graph)
    
    pairs = []
    for node in graph.node:
        if node.op_type == "MatMul":
            # Check if this is 3D
            is_3d = False
            for inp in node.input:
                if inp in shapes and len(shapes[inp]) >= 3:
                    is_3d = True
                    break
            
            if not is_3d:
                continue
            
            # Check if output goes to Add
            for out in node.output:
                if out in input_map:
                    for consumer in input_map[out]:
                        if consumer.op_type == "Add":
                            # Check if other input is initializer (bias)
                            other_inputs = [i for i in consumer.input if i != out]
                            bias_is_init = any(i in inits for i in other_inputs)
                            pairs.append({
                                "matmul_name": node.name,
                                "add_name": consumer.name,
                                "matmul_output": out,
                                "bias_is_initializer": bias_is_init
                            })
    return pairs


def find_identity_transposes(graph) -> List[Dict]:
    """Find Transpose nodes with identity permutation."""
    results = []
    for node in graph.node:
        if node.op_type == "Transpose":
            perm = None
            for attr in node.attribute:
                if attr.name == "perm":
                    perm = list(attr.ints)
                    break
            
            if perm is not None:
                # Check if perm is identity [0, 1, 2, ...n]
                if perm == list(range(len(perm))):
                    results.append({
                        "node_name": node.name,
                        "perm": perm,
                        "is_identity": True
                    })
    return results


def find_identity_reshapes(graph) -> List[Dict]:
    """Find Reshape nodes where output shape equals input shape."""
    shapes = get_value_info_shapes(graph)
    inits = get_initializer_names(graph)
    
    results = []
    for node in graph.node:
        if node.op_type == "Reshape":
            if len(node.input) >= 1 and len(node.output) >= 1:
                inp = node.input[0]
                out = node.output[0]
                
                if inp in shapes and out in shapes:
                    in_shape = shapes[inp]
                    out_shape = shapes[out]
                    
                    # Check if shapes are identical (ignoring dynamic dims)
                    if in_shape == out_shape and -1 not in in_shape:
                        results.append({
                            "node_name": node.name,
                            "input_shape": in_shape,
                            "output_shape": out_shape
                        })
    return results


def find_consecutive_ops(graph, op_type: str) -> List[Dict]:
    """Find chains of consecutive same-type operations."""
    output_map = build_node_output_map(graph)
    input_map = build_node_input_map(graph)
    
    chains = []
    visited = set()
    
    for node in graph.node:
        if node.op_type != op_type or node.name in visited:
            continue
        
        # Build chain forward
        chain = [node.name]
        visited.add(node.name)
        current = node
        
        while True:
            # Find next node of same type
            found_next = False
            for out in current.output:
                if out in input_map:
                    for consumer in input_map[out]:
                        if consumer.op_type == op_type and consumer.name not in visited:
                            chain.append(consumer.name)
                            visited.add(consumer.name)
                            current = consumer
                            found_next = True
                            break
                if found_next:
                    break
            if not found_next:
                break
        
        if len(chain) > 1:
            chains.append({
                "op_type": op_type,
                "chain_length": len(chain),
                "nodes": chain
            })
    
    return chains


def find_cast_chains(graph) -> List[Dict]:
    """Find Cast → Cast chains."""
    return find_consecutive_ops(graph, "Cast")


def find_redundant_casts(graph) -> List[Dict]:
    """Find Cast nodes that cast to the same dtype as input."""
    shapes = get_value_info_shapes(graph)
    results = []
    
    # Get value info types
    type_map = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.type.HasField('tensor_type'):
            type_map[vi.name] = vi.type.tensor_type.elem_type
    
    for node in graph.node:
        if node.op_type == "Cast":
            to_type = None
            for attr in node.attribute:
                if attr.name == "to":
                    to_type = attr.i
                    break
            
            if to_type and len(node.input) > 0:
                inp = node.input[0]
                if inp in type_map and type_map[inp] == to_type:
                    results.append({
                        "node_name": node.name,
                        "input": inp,
                        "to_type": to_type,
                        "is_redundant": True
                    })
    
    return results


def find_qkv_triple_matmul(graph) -> List[Dict]:
    """Find QKV attention pattern: 3 MatMuls with same input, different weights."""
    output_map = build_node_output_map(graph)
    inits = get_initializer_names(graph)
    
    # Group MatMuls by their data input (non-weight)
    matmul_by_input = defaultdict(list)
    
    for node in graph.node:
        if node.op_type == "MatMul" and len(node.input) == 2:
            inp_a, inp_b = node.input
            
            # Determine which is data and which is weight
            a_is_init = inp_a in inits
            b_is_init = inp_b in inits
            
            if a_is_init and not b_is_init:
                data_input = inp_b
                weight_input = inp_a
            elif b_is_init and not a_is_init:
                data_input = inp_a
                weight_input = inp_b
            else:
                continue
            
            matmul_by_input[data_input].append({
                "node": node,
                "weight": weight_input
            })
    
    # Find groups of 3 (Q, K, V)
    qkv_patterns = []
    for data_input, matmuls in matmul_by_input.items():
        if len(matmuls) >= 3:
            qkv_patterns.append({
                "data_input": data_input,
                "matmul_count": len(matmuls),
                "matmul_names": [m["node"].name for m in matmuls],
                "weights": [m["weight"] for m in matmuls]
            })
    
    return qkv_patterns


def find_layernorm_ops(graph) -> List[Dict]:
    """Find LayerNormalization ops (opset 17+)."""
    results = []
    for node in graph.node:
        if node.op_type in ["LayerNormalization", "LayerNorm"]:
            results.append({
                "node_name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input)
            })
    return results


def find_layernorm_subgraph(graph) -> List[Dict]:
    """Find decomposed LayerNorm pattern: ReduceMean → Sub → Pow → ReduceMean → Add → Sqrt → Div."""
    output_map = build_node_output_map(graph)
    input_map = build_node_input_map(graph)
    
    # Look for ReduceMean → Sub patterns as starting point
    patterns = []
    for node in graph.node:
        if node.op_type == "ReduceMean":
            # Check if output feeds into Sub
            for out in node.output:
                if out in input_map:
                    for consumer in input_map[out]:
                        if consumer.op_type == "Sub":
                            # Potential LayerNorm start
                            patterns.append({
                                "reduce_mean_node": node.name,
                                "sub_node": consumer.name,
                                "likely_layernorm": True
                            })
    return patterns


def find_expand_identity(graph) -> List[Dict]:
    """Find Expand ops where output shape equals input shape."""
    shapes = get_value_info_shapes(graph)
    
    results = []
    for node in graph.node:
        if node.op_type == "Expand":
            if len(node.input) >= 1 and len(node.output) >= 1:
                inp = node.input[0]
                out = node.output[0]
                
                if inp in shapes and out in shapes:
                    in_shape = shapes[inp]
                    out_shape = shapes[out]
                    
                    if in_shape == out_shape and -1 not in in_shape:
                        results.append({
                            "node_name": node.name,
                            "input_shape": in_shape,
                            "output_shape": out_shape
                        })
    return results


def find_shape_gather_unsqueeze_concat(graph) -> List[Dict]:
    """Find Shape → Gather → Unsqueeze → Concat chains (shape computation)."""
    output_map = build_node_output_map(graph)
    input_map = build_node_input_map(graph)
    
    patterns = []
    for node in graph.node:
        if node.op_type == "Shape":
            # Follow the chain
            chain = [node.name]
            current_outputs = list(node.output)
            
            for _ in range(5):  # Max depth
                next_outputs = []
                for out in current_outputs:
                    if out in input_map:
                        for consumer in input_map[out]:
                            if consumer.op_type in ["Gather", "Unsqueeze", "Concat", "Squeeze"]:
                                chain.append(f"{consumer.op_type}:{consumer.name}")
                                next_outputs.extend(consumer.output)
                current_outputs = next_outputs
                if not current_outputs:
                    break
            
            if len(chain) > 1:
                patterns.append({
                    "shape_node": node.name,
                    "chain": chain,
                    "chain_length": len(chain)
                })
    
    return patterns


def find_constant_subgraphs(graph) -> List[Dict]:
    """Find subgraphs where all inputs are constants (could be folded)."""
    inits = get_initializer_names(graph)
    graph_inputs = {inp.name for inp in graph.input}
    
    # Node outputs that are constant
    constant_outputs = set(inits)
    
    # Iteratively find nodes with all-constant inputs
    foldable = []
    changed = True
    while changed:
        changed = False
        for node in graph.node:
            if node.name in [f["node_name"] for f in foldable]:
                continue
            
            # Skip nodes that take graph inputs (runtime values)
            has_runtime_input = any(inp in graph_inputs for inp in node.input)
            if has_runtime_input:
                continue
            
            # Check if all inputs are constant
            all_const = all(
                inp in constant_outputs or inp == ""
                for inp in node.input
            )
            
            if all_const and node.op_type not in ["Constant", "Shape"]:
                foldable.append({
                    "node_name": node.name,
                    "op_type": node.op_type,
                    "inputs": list(node.input)
                })
                constant_outputs.update(node.output)
                changed = True
    
    return foldable


def get_op_histogram(graph) -> Dict[str, int]:
    """Get histogram of op types."""
    hist = defaultdict(int)
    for node in graph.node:
        hist[node.op_type] += 1
    return dict(sorted(hist.items(), key=lambda x: -x[1]))


def audit_transformer(model_path: str) -> Dict[str, Any]:
    """Run complete forensic audit on a model."""
    print(f"\n{'='*60}")
    print(f"FORENSIC AUDIT: {model_path}")
    print(f"{'='*60}")
    
    model = onnx.load(model_path)
    graph = model.graph
    
    report = {
        "model_path": model_path,
        "total_nodes": len(graph.node),
        "total_initializers": len(graph.initializer),
        "op_histogram": get_op_histogram(graph),
        
        # Core findings
        "matmul_3d": find_3d_matmuls(graph),
        "matmul_add_3d_pairs": find_3d_matmul_add_pairs(graph),
        "identity_transposes": find_identity_transposes(graph),
        "identity_reshapes": find_identity_reshapes(graph),
        "consecutive_reshapes": find_consecutive_ops(graph, "Reshape"),
        "consecutive_transposes": find_consecutive_ops(graph, "Transpose"),
        "cast_chains": find_cast_chains(graph),
        "redundant_casts": find_redundant_casts(graph),
        "qkv_patterns": find_qkv_triple_matmul(graph),
        "layernorm_ops": find_layernorm_ops(graph),
        "layernorm_subgraph": find_layernorm_subgraph(graph),
        "expand_identity": find_expand_identity(graph),
        "shape_chains": find_shape_gather_unsqueeze_concat(graph),
        "unfused_constants": find_constant_subgraphs(graph),
    }
    
    return report


def print_report(report: Dict[str, Any]):
    """Pretty-print the audit report."""
    print(f"\nTotal nodes: {report['total_nodes']}")
    print(f"Total initializers: {report['total_initializers']}")
    
    print(f"\n--- Op Histogram (top 15) ---")
    for i, (op, count) in enumerate(report['op_histogram'].items()):
        if i >= 15:
            break
        print(f"  {op:25s} {count:4d}")
    
    print(f"\n--- 3D MatMul Nodes ---")
    print(f"  Count: {len(report['matmul_3d'])}")
    
    print(f"\n--- 3D MatMul+Add Pairs (Known Gap) ---")
    print(f"  Count: {len(report['matmul_add_3d_pairs'])}")
    for p in report['matmul_add_3d_pairs'][:5]:
        print(f"    MatMul: {p['matmul_name'][:40]} → Add: {p['add_name'][:30]}")
    if len(report['matmul_add_3d_pairs']) > 5:
        print(f"    ... and {len(report['matmul_add_3d_pairs'])-5} more")
    
    print(f"\n--- Identity Transposes (Noop) ---")
    print(f"  Count: {len(report['identity_transposes'])}")
    for t in report['identity_transposes'][:3]:
        print(f"    {t['node_name']} perm={t['perm']}")
    
    print(f"\n--- Identity Reshapes (Noop) ---")
    print(f"  Count: {len(report['identity_reshapes'])}")
    for r in report['identity_reshapes'][:3]:
        print(f"    {r['node_name']} shape={r['input_shape']}")
    
    print(f"\n--- Consecutive Reshape Chains ---")
    print(f"  Count: {len(report['consecutive_reshapes'])}")
    for c in report['consecutive_reshapes'][:3]:
        print(f"    Length {c['chain_length']}: {c['nodes'][:3]}...")
    
    print(f"\n--- Consecutive Transpose Chains ---")
    print(f"  Count: {len(report['consecutive_transposes'])}")
    for c in report['consecutive_transposes'][:3]:
        print(f"    Length {c['chain_length']}: {c['nodes'][:3]}...")
    
    print(f"\n--- Cast Chains ---")
    print(f"  Count: {len(report['cast_chains'])}")
    
    print(f"\n--- Redundant Casts (same type) ---")
    print(f"  Count: {len(report['redundant_casts'])}")
    
    print(f"\n--- QKV Triple-MatMul Patterns ---")
    print(f"  Count: {len(report['qkv_patterns'])}")
    for q in report['qkv_patterns'][:3]:
        print(f"    Input: {q['data_input'][:30]} → {q['matmul_count']} MatMuls")
    
    print(f"\n--- LayerNorm Ops (native) ---")
    print(f"  Count: {len(report['layernorm_ops'])}")
    
    print(f"\n--- LayerNorm Subgraph (decomposed) ---")
    print(f"  Count: {len(report['layernorm_subgraph'])}")
    
    print(f"\n--- Expand Identity ---")
    print(f"  Count: {len(report['expand_identity'])}")
    
    print(f"\n--- Shape Computation Chains ---")
    print(f"  Count: {len(report['shape_chains'])}")
    
    print(f"\n--- Unfused Constant Subgraphs ---")
    print(f"  Count: {len(report['unfused_constants'])}")
    for u in report['unfused_constants'][:5]:
        print(f"    {u['op_type']}: {u['node_name'][:40]}")


def print_summary_table(reports: List[Dict[str, Any]]):
    """Print comparative summary table across all models."""
    print(f"\n{'='*100}")
    print("FORENSICS SUMMARY TABLE")
    print(f"{'='*100}")
    
    # Extract model names
    names = [r['model_path'].split('/')[-1].replace('.onnx', '') for r in reports]
    
    # Header
    print(f"{'Finding':<35}", end="")
    for name in names:
        print(f" | {name[:12]:>12}", end="")
    print()
    print("-" * 100)
    
    # Rows
    findings = [
        ("Total nodes", lambda r: r['total_nodes']),
        ("3D MatMul+Add pairs", lambda r: len(r['matmul_add_3d_pairs'])),
        ("Identity transposes", lambda r: len(r['identity_transposes'])),
        ("Identity reshapes", lambda r: len(r['identity_reshapes'])),
        ("Consecutive reshape chains", lambda r: len(r['consecutive_reshapes'])),
        ("Consecutive transpose chains", lambda r: len(r['consecutive_transposes'])),
        ("Cast chains", lambda r: len(r['cast_chains'])),
        ("Redundant casts", lambda r: len(r['redundant_casts'])),
        ("QKV triple-MatMul patterns", lambda r: len(r['qkv_patterns'])),
        ("LayerNorm ops (native)", lambda r: len(r['layernorm_ops'])),
        ("LayerNorm subgraph", lambda r: len(r['layernorm_subgraph'])),
        ("Expand identity", lambda r: len(r['expand_identity'])),
        ("Shape chains", lambda r: len(r['shape_chains'])),
        ("Unfused constants", lambda r: len(r['unfused_constants'])),
    ]
    
    for finding_name, extractor in findings:
        print(f"{finding_name:<35}", end="")
        for r in reports:
            val = extractor(r)
            print(f" | {val:>12}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="Graph forensics for transformer ONNX models")
    parser.add_argument("models", nargs="+", help="ONNX model files to audit")
    parser.add_argument("--summary", action="store_true", help="Print summary table only")
    args = parser.parse_args()
    
    reports = []
    for model_path in args.models:
        try:
            report = audit_transformer(model_path)
            reports.append(report)
            if not args.summary:
                print_report(report)
        except Exception as e:
            print(f"ERROR auditing {model_path}: {e}")
    
    if len(reports) > 1 or args.summary:
        print_summary_table(reports)


if __name__ == "__main__":
    main()
