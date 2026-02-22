"""
Script to export EfficientNet-B0 to ONNX
"""
import torchvision
import torch

model = torchvision.models.efficientnet_b0(weights='DEFAULT').eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'models/efficientnet-b0.onnx',
    opset_version=13, input_names=['input'], output_names=['output'],
    do_constant_folding=False)
print('Done: models/efficientnet-b0.onnx')
