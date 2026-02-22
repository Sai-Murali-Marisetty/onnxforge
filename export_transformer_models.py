#!/usr/bin/env python3
"""
Export Transformer Models for M10 Benchmark Suite

Models exported:
- BERT-base-uncased (12 layers, 110M params)
- DistilBERT-base-uncased (6 layers, 66M params)
- RoBERTa-base (12 layers, 125M params)

These are exported with do_constant_folding=False to preserve
optimization opportunities for onnxslim.
"""

import os
import torch
from transformers import (
    BertModel, BertTokenizer,
    DistilBertModel, DistilBertTokenizer,
    RobertaModel, RobertaTokenizer,
)

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

print("=" * 60)
print("M10 Transformer Model Export")
print("=" * 60)

# Common export settings
SEQ_LEN = 128
BATCH_SIZE = 1
OPSET_VERSION = 14

def export_bert_model(model_class, tokenizer_class, model_name, output_name, has_token_type=True):
    """Export a BERT-family model to ONNX."""
    print(f"\nExporting {model_name}...")
    
    model = model_class.from_pretrained(model_name).eval()
    
    # Create dummy inputs
    dummy_ids = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    dummy_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    
    # Prepare inputs based on model type
    if has_token_type:
        dummy_token_type = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
        inputs = (dummy_ids, dummy_mask, dummy_token_type)
        input_names = ['input_ids', 'attention_mask', 'token_type_ids']
        dynamic_axes = {
            'input_ids': {0: 'batch', 1: 'seq'},
            'attention_mask': {0: 'batch', 1: 'seq'},
            'token_type_ids': {0: 'batch', 1: 'seq'},
        }
    else:
        inputs = (dummy_ids, dummy_mask)
        input_names = ['input_ids', 'attention_mask']
        dynamic_axes = {
            'input_ids': {0: 'batch', 1: 'seq'},
            'attention_mask': {0: 'batch', 1: 'seq'},
        }
    
    output_path = f'models/{output_name}.onnx'
    
    torch.onnx.export(
        model,
        inputs,
        output_path,
        opset_version=OPSET_VERSION,
        do_constant_folding=False,  # Preserve optimization opportunities
        input_names=input_names,
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes={
            **dynamic_axes,
            'last_hidden_state': {0: 'batch', 1: 'seq'},
            'pooler_output': {0: 'batch'},
        }
    )
    
    print(f"  -> {output_path}")
    return output_path

# Export models
models_to_export = [
    (BertModel, BertTokenizer, 'bert-base-uncased', 'bert_base', True),
    (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased', 'distilbert_base', False),
    (RobertaModel, RobertaTokenizer, 'roberta-base', 'roberta_base', True),
]

exported_paths = []
for model_cls, tok_cls, name, out_name, has_tt in models_to_export:
    try:
        path = export_bert_model(model_cls, tok_cls, name, out_name, has_tt)
        exported_paths.append(path)
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "=" * 60)
print("Transformer model export complete!")
print("=" * 60)

# Verify exports
import onnx

for model_path in exported_paths:
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print(f"\n{model_path}:")
        print(f"  Nodes: {len(model.graph.node)}")
        print(f"  Inputs: {[i.name for i in model.graph.input]}")
        print(f"  Outputs: {[o.name for o in model.graph.output]}")
    except Exception as e:
        print(f"\n{model_path}: Error - {e}")
