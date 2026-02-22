"""
M11 Latency Benchmark - Exp 20
Measure ORT CPU latency before and after optimization for all models.
"""
import onnxruntime as ort
import numpy as np
import time
import os

def benchmark_model(model_path, input_feed, n_warmup=5, n_runs=20):
    """Benchmark a single model."""
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Warmup
    for _ in range(n_warmup):
        sess.run(None, input_feed)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        sess.run(None, input_feed)
        times.append((time.perf_counter() - start) * 1000)  # ms
    
    return np.mean(times), np.std(times)

def get_input_feed(model_name):
    """Generate appropriate input for each model type."""
    if 'roberta' in model_name:
        # RoBERTa doesn't use token_type_ids
        return {
            'input_ids': np.ones((1, 128), dtype=np.int64),
            'attention_mask': np.ones((1, 128), dtype=np.int64),
            'token_type_ids': np.zeros((1, 128), dtype=np.int64)
        }
    elif 'distilbert' in model_name:
        # DistilBERT doesn't use token_type_ids
        return {
            'input_ids': np.ones((1, 128), dtype=np.int64),
            'attention_mask': np.ones((1, 128), dtype=np.int64)
        }
    elif 'bert' in model_name:
        return {
            'input_ids': np.ones((1, 128), dtype=np.int64),
            'attention_mask': np.ones((1, 128), dtype=np.int64),
            'token_type_ids': np.zeros((1, 128), dtype=np.int64)
        }
    elif 'whisper' in model_name:
        return {'input_features': np.random.randn(1, 80, 3000).astype(np.float32)}
    elif 'deit' in model_name or 'vit' in model_name:
        return {'pixel_values': np.random.randn(1, 3, 224, 224).astype(np.float32)}
    elif 'yolo' in model_name:
        return {'images': np.random.randn(1, 3, 640, 640).astype(np.float32)}
    elif 'efficientnet' in model_name or 'resnet' in model_name or 'mobilenet' in model_name:
        return {'data': np.random.randn(1, 3, 224, 224).astype(np.float32)}
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def main():
    models_dir = 'models'
    
    # Model pairs: (original, optimized, name)
    model_pairs = [
        ('bert_base.onnx', 'bert_base_m11.onnx', 'BERT-base'),
        ('distilbert_base.onnx', 'distilbert_m11.onnx', 'DistilBERT'),
        ('roberta_base.onnx', 'roberta_m11.onnx', 'RoBERTa'),
        ('whisper_tiny_encoder.onnx', 'whisper_m11.onnx', 'Whisper-tiny'),
        ('whisper_base_encoder.onnx', 'whisper_base_opt.onnx', 'Whisper-base'),
        ('deit_small.onnx', 'deit_small_opt.onnx', 'DeiT-Small'),
        ('vit_base.onnx', 'vit_base_opt.onnx', 'ViT-Base'),
    ]
    
    print("=" * 80)
    print("M11 LATENCY BENCHMARK (ORT CPU)")
    print("=" * 80)
    print(f"{'Model':<20} {'Before (ms)':<15} {'After (ms)':<15} {'Speedup':<10} {'Change':<10}")
    print("-" * 80)
    
    results = []
    for orig_file, opt_file, name in model_pairs:
        orig_path = os.path.join(models_dir, orig_file)
        opt_path = os.path.join(models_dir, opt_file)
        
        if not os.path.exists(orig_path) or not os.path.exists(opt_path):
            print(f"{name:<20} SKIPPED (file not found)")
            continue
        
        try:
            input_feed = get_input_feed(name.lower())
            
            orig_mean, orig_std = benchmark_model(orig_path, input_feed)
            opt_mean, opt_std = benchmark_model(opt_path, input_feed)
            
            speedup = orig_mean / opt_mean
            change_pct = (orig_mean - opt_mean) / orig_mean * 100
            
            print(f"{name:<20} {orig_mean:>8.2f} ± {orig_std:>4.1f}  {opt_mean:>8.2f} ± {opt_std:>4.1f}  {speedup:>6.3f}x   {change_pct:>+6.1f}%")
            
            results.append({
                'name': name,
                'before': orig_mean,
                'after': opt_mean,
                'speedup': speedup,
                'change_pct': change_pct
            })
        except Exception as e:
            print(f"{name:<20} ERROR: {e}")
    
    print("-" * 80)
    
    if results:
        avg_speedup = np.mean([r['speedup'] for r in results])
        avg_change = np.mean([r['change_pct'] for r in results])
        print(f"{'AVERAGE':<20} {'':<15} {'':<15} {avg_speedup:>6.3f}x   {avg_change:>+6.1f}%")

if __name__ == "__main__":
    main()
