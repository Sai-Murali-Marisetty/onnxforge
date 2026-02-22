#!/usr/bin/env python3
"""
Export DistilBERT and Whisper for M10 Benchmark Suite

- DistilBERT-base-uncased (6 layers, 66M params) - no pooler output
- Whisper-tiny encoder (audio model)
"""

import os
import torch
from transformers import (
    DistilBertModel,
    WhisperForConditionalGeneration,
)

os.makedirs('models', exist_ok=True)

print("=" * 60)
print("M10 Additional Transformer Export")
print("=" * 60)

SEQ_LEN = 128
BATCH_SIZE = 1
OPSET_VERSION = 14

# DistilBERT (no pooler, only 1 output)
print("\n[1/2] Exporting DistilBERT-base-uncased...")
model = DistilBertModel.from_pretrained('distilbert-base-uncased').eval()

dummy_ids = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
dummy_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)

torch.onnx.export(
    model,
    (dummy_ids, dummy_mask),
    'models/distilbert_base.onnx',
    opset_version=OPSET_VERSION,
    do_constant_folding=False,
    input_names=['input_ids', 'attention_mask'],
    output_names=['last_hidden_state'],  # DistilBERT has no pooler
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'seq'},
        'attention_mask': {0: 'batch', 1: 'seq'},
        'last_hidden_state': {0: 'batch', 1: 'seq'},
    }
)
print("  -> models/distilbert_base.onnx")

# Whisper-tiny encoder
print("\n[2/2] Exporting Whisper-tiny encoder...")
whisper_model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny').eval()
encoder = whisper_model.model.encoder

# Whisper expects mel spectrogram: [batch, 80, 3000] for 30s audio
dummy_input = torch.randn(1, 80, 3000)

torch.onnx.export(
    encoder,
    dummy_input,
    'models/whisper_tiny_encoder.onnx',
    opset_version=OPSET_VERSION,
    do_constant_folding=False,
    input_names=['input_features'],
    output_names=['last_hidden_state'],
)
print("  -> models/whisper_tiny_encoder.onnx")

print("\n" + "=" * 60)
print("Export complete!")
print("=" * 60)

# Verify exports
import onnx

for model_path in ['models/distilbert_base.onnx', 'models/whisper_tiny_encoder.onnx']:
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print(f"\n{model_path}:")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Inputs: {[i.name for i in model.graph.input]}")
    print(f"  Outputs: {[o.name for o in model.graph.output]}")
