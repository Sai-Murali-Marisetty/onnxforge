#!/usr/bin/env python3
"""
Export Vision Models for M10 Benchmark Suite

Models exported:
- ResNet-50 (torchvision)
- MobileNetV3-Small (torchvision)

These are exported with do_constant_folding=False to preserve 
optimization opportunities for onnxslim.
"""

import os
import torch
import torchvision.models as models

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

print("=" * 60)
print("M10 Vision Model Export")
print("=" * 60)

# ResNet-50
print("\n[1/2] Exporting ResNet-50...")
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    resnet50,
    dummy_input,
    'models/resnet50.onnx',
    opset_version=12,
    do_constant_folding=False,  # Preserve optimization opportunities
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None,  # Fixed shapes for simpler benchmarking
)
print("  -> models/resnet50.onnx")

# MobileNetV3-Small
print("\n[2/2] Exporting MobileNetV3-Small...")
mobilenetv3 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()

torch.onnx.export(
    mobilenetv3,
    dummy_input,
    'models/mobilenetv3_small.onnx',
    opset_version=12,
    do_constant_folding=False,  # Preserve optimization opportunities
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None,
)
print("  -> models/mobilenetv3_small.onnx")

print("\n" + "=" * 60)
print("Vision model export complete!")
print("=" * 60)

# Verify exports
import onnx

for model_path in ['models/resnet50.onnx', 'models/mobilenetv3_small.onnx']:
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print(f"\n{model_path}:")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Inputs: {[i.name for i in model.graph.input]}")
    print(f"  Outputs: {[o.name for o in model.graph.output]}")
