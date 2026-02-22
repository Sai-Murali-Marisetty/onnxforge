#!/usr/bin/env python3
"""
Exp 25 — Pre-optimization Contamination Study

Question: Is YOLOv8n pre-optimized by Ultralytics' export pipeline?
Compare: raw PyTorch export vs. Ultralytics hub export vs. onnxforge output
"""

import os
import torch
import onnx
from collections import Counter

def export_yolov8n_raw():
    """Export YOLOv8n using raw torch.onnx without Ultralytics preprocessing."""
    from ultralytics import YOLO
    
    # Load pretrained model
    model = YOLO('yolov8n.pt')
    pytorch_model = model.model.eval()
    
    print(f"YOLOv8n PyTorch model type: {type(pytorch_model)}")
    print(f"Parameters: {sum(p.numel() for p in pytorch_model.parameters())/1e6:.1f}M")
    
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # Export WITHOUT constant folding
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        'models/yolov8n_raw.onnx',
        opset_version=12,
        do_constant_folding=False,
        input_names=['images'],
        output_names=['output']
    )
    print("RAW export saved to models/yolov8n_raw.onnx")
    return 'models/yolov8n_raw.onnx'

def analyze_model(path, name):
    """Analyze an ONNX model."""
    if not os.path.exists(path):
        print(f"{name}: NOT FOUND")
        return None
    
    model = onnx.load(path)
    graph = model.graph
    
    ops = Counter(n.op_type for n in graph.node)
    
    print(f"\n{name}:")
    print(f"  Total nodes: {len(graph.node)}")
    print(f"  Initializers: {len(graph.initializer)}")
    print(f"  Top ops: {ops.most_common(5)}")
    
    # Count specific optimization indicators
    identity_count = ops.get('Identity', 0)
    constant_count = ops.get('Constant', 0)
    transpose_count = ops.get('Transpose', 0)
    
    print(f"  Identity: {identity_count}, Constant: {constant_count}, Transpose: {transpose_count}")
    
    return {
        'nodes': len(graph.node),
        'initializers': len(graph.initializer),
        'ops': ops
    }

def main():
    print("=" * 70)
    print("Exp 25: Pre-optimization Contamination Study")
    print("=" * 70)
    print("""
Question: How much optimization does Ultralytics' export pipeline apply?
If YOLOv8n from the hub is already optimized, our 0% reduction is expected.
""")
    
    # Check existing hub export
    print("\n1. Analyzing existing hub export (yolov8n.onnx)...")
    hub_analysis = analyze_model('models/yolov8n.onnx', 'YOLOv8n (Hub Export)')
    
    # Try raw export
    print("\n2. Attempting raw PyTorch export...")
    try:
        raw_path = export_yolov8n_raw()
        raw_analysis = analyze_model(raw_path, 'YOLOv8n (Raw Export)')
    except Exception as e:
        print(f"  Raw export failed: {type(e).__name__}: {e}")
        raw_analysis = None
    
    # Compare
    if hub_analysis and raw_analysis:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        print(f"\n  Hub export:  {hub_analysis['nodes']} nodes")
        print(f"  Raw export:  {raw_analysis['nodes']} nodes")
        diff = raw_analysis['nodes'] - hub_analysis['nodes']
        print(f"  Difference:  {diff} nodes ({diff/raw_analysis['nodes']*100:.1f}% reduction by Ultralytics)")
        
        if diff > 0:
            print(f"\n  CONCLUSION: Ultralytics applies {diff} nodes of optimization")
            print(f"  Our 0% reduction on hub export is expected - already optimized")
        else:
            print(f"\n  CONCLUSION: Hub export has MORE nodes than raw export")
            print(f"  Ultralytics may add metadata nodes")
    
    print("\n" + "=" * 70)
    print("IMPLICATIONS")
    print("=" * 70)
    print("""
If hub models are pre-optimized:
1. Our benchmarks should use RAW exports as baseline
2. 0% reduction on YOLOv8n is correct - nothing left to optimize
3. Need to clearly document which models are pre-optimized

For fair comparison:
- HuggingFace transformers: use do_constant_folding=False
- Ultralytics YOLO: use raw torch.onnx.export, not YOLO.export()
- ONNX Model Zoo: may already be optimized by contributors
""")

if __name__ == "__main__":
    main()
