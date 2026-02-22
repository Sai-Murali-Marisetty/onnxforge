#!/usr/bin/env python3
"""Quick check if RoBERTa 11.8% is measurement variance."""

import numpy as np
import onnxruntime as ort
import time

def measure_latency(path, name, n_runs=100):
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    inputs = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        if 'input_ids' in inp.name:
            shape = [1, 128]
        elif 'attention_mask' in inp.name:
            shape = [1, 128]
        elif 'token_type_ids' in inp.name:
            shape = [1, 128]
            inputs[inp.name] = np.zeros(shape, dtype=np.int64)
            continue
        
        if inp.type == 'tensor(int64)':
            inputs[inp.name] = np.ones(shape, dtype=np.int64)
        else:
            inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        sess.run(None, inputs)
    
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        sess.run(None, inputs)
        latencies.append((time.perf_counter() - start) * 1000)
    
    return np.mean(latencies), np.std(latencies)

print('Extended latency measurement (100 runs each):')
print('-' * 60)

bert_mean, bert_std = measure_latency('models/bert_base.onnx', 'BERT')
print(f'BERT (original):    {bert_mean:.2f} +/- {bert_std:.2f} ms')

roberta_mean, roberta_std = measure_latency('models/roberta_base.onnx', 'RoBERTa')
print(f'RoBERTa (original): {roberta_mean:.2f} +/- {roberta_std:.2f} ms')

print()
print(f'RoBERTa variance: std/mean = {roberta_std/roberta_mean*100:.1f}%')
print(f'BERT variance: std/mean = {bert_std/bert_mean*100:.1f}%')
print()
print(f'If RoBERTa speedup is 11.8%, that is {11.8 / (roberta_std/roberta_mean*100):.1f}x the std')
print(f'For BERT speedup of 1.7%, that is {1.7 / (bert_std/bert_mean*100):.1f}x the std')
print()

# The key insight: if variance is ~1%, then 11.8% is 11x the variance
# That would be statistically significant

# Additional check: are the baseline latencies similar?
print(f'Baseline latency difference: {(roberta_mean - bert_mean):.1f} ms')
print(f'Relative difference: {(roberta_mean - bert_mean)/bert_mean*100:.1f}%')
