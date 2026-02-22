"""
M11 Pass Attribution Matrix - Exp 19
Measure node reduction from each pass in isolation.
"""
import onnx
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    FuseMatmulAdd3d,
    CleanupAttention,
)

def count_nodes(model):
    return len(model.graph.node)

def test_pass_isolated(model_path, pass_instance):
    """Run a single pass and measure node reduction."""
    model = onnx.load(model_path)
    before = count_nodes(model)
    
    try:
        model = pass_instance.run(model)
        after = count_nodes(model)
        return before - after
    except Exception as e:
        return f"ERR"

def main():
    models_dir = 'models'
    
    # Models to test
    models = [
        ('resnet50.onnx', 'ResNet-50'),
        ('efficientnet-b0.onnx', 'EfficientNet'),
        ('mobilenetv3_small.onnx', 'MobileNetV3'),
        ('bert_base.onnx', 'BERT-base'),
        ('distilbert_base.onnx', 'DistilBERT'),
        ('roberta_base.onnx', 'RoBERTa'),
        ('whisper_tiny_encoder.onnx', 'Whisper-tiny'),
        ('whisper_base_encoder.onnx', 'Whisper-base'),
        ('deit_small.onnx', 'DeiT-Small'),
        ('vit_base.onnx', 'ViT-Base'),
        ('yolov8n.onnx', 'YOLOv8n'),
    ]
    
    # Passes to test
    passes = [
        (EliminateDeadNodes(), 'dead_nodes'),
        (EliminateIdentityOps(), 'identity'),
        (EliminateUnusedInitializers(), 'unused_init'),
        (EliminateDuplicateConstants(), 'dup_const'),
        (EliminateRedundantTransposes(), 'red_trans'),
        (FoldConstants(), 'fold_const'),
        (SimplifyShapeChains(), 'shape_chains'),
        (FuseConvBatchnorm(), 'conv_bn'),
        (FuseConvRelu(), 'conv_relu'),
        (FuseMatmulAdd(), 'mm_add'),
        (FuseMatmulAdd3d(), 'mm_add_3d'),
        (CleanupAttention(), 'attn'),
    ]
    
    # Header
    pass_names = [p[1] for p in passes]
    header = f"{'Model':<14}" + "".join([f" {n:>10}" for n in pass_names]) + "  TOTAL"
    print("=" * len(header))
    print("PASS ATTRIBUTION MATRIX (Δ nodes, isolated runs)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    
    for model_file, model_name in models:
        model_path = os.path.join(models_dir, model_file)
        if not os.path.exists(model_path):
            continue
        
        row = f"{model_name:<14}"
        total = 0
        
        for pass_instance, pass_short in passes:
            delta = test_pass_isolated(model_path, pass_instance)
            if isinstance(delta, int):
                row += f" {delta:>10}"
                total += delta
            else:
                row += f" {delta:>10}"
        
        row += f"  {total:>5}"
        print(row)
    
    print("-" * len(header))

if __name__ == "__main__":
    main()
