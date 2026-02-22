# Micro-plan M10 — The Breakthrough Milestone

## Milestone Goal

Prove every pass on real production models. Fill every blank in the attribution table.
Build the complete benchmark suite that anchors the research agenda.
This is the milestone that turns onnxslim from "a tool I built" into "a credible optimizer
with published numbers across 8 models, 11 passes, and 3 conversion targets."

M10 is not done until every model is tested, every experiment has real numbers,
and the README benchmark table has no TBDs.

---

## Status: 🔴 NOT STARTED

---

## Models to Download and Test

### Group A — Vision Models (Transposes + BN Fusion)

| Model | Source | Why |
|-------|--------|-----|
| YOLOv8n | `ultralytics` export | Transpose hell — proves M4 fires on real vision model |
| YOLOv8s | `ultralytics` export | Larger YOLO — confirms scaling behavior |
| EfficientDet-D0 | ONNX Model Zoo | Different BN pattern than EfficientNet |
| ResNet-50 | ONNX Model Zoo / torchvision | Classic baseline — everyone benchmarks against it |
| MobileNetV3-Small | torchvision export | Successor to V2, different op profile |

### Group B — Transformer Models (The Core Proof)

| Model | Source | Why |
|-------|--------|-----|
| BERT-base-uncased | HuggingFace `transformers` | Primary gate — proves attention, shape chains, matmul+add |
| BERT-large-uncased | HuggingFace `transformers` | Scale test — does reduction % hold at 2x size? |
| DistilBERT-base | HuggingFace `transformers` | Distilled variant — different attention pattern |
| RoBERTa-base | HuggingFace `transformers` | BERT variant — slightly different export signature |
| Whisper-tiny | HuggingFace `transformers` | Encoder-decoder — cross-attention variant |
| Whisper-base | HuggingFace `transformers` | Scale test above tiny |
| ALBERT-base-v2 | HuggingFace `transformers` | Parameter sharing — unique graph structure |

### Group C — Hybrid / Deployment Models

| Model | Source | Why |
|-------|--------|-----|
| MobileViT-Small | HuggingFace | Hybrid CNN+Transformer — tests both pass families |
| DeiT-Small | HuggingFace | Pure vision transformer — no Conv, pure MatMul |
| Phi-2 | HuggingFace (if feasible) | LLM — biggest differentiator demo |

---

## Download Scripts

### Group A — Vision

```bash
# YOLOv8n + YOLOv8s (requires ultralytics)
pip install ultralytics
python -c "
from ultralytics import YOLO
YOLO('yolov8n.pt').export(format='onnx', opset=12, dynamic=False)
YOLO('yolov8s.pt').export(format='onnx', opset=12, dynamic=False)
"

# ResNet-50
python -c "
import torch, torchvision
model = torchvision.models.resnet50(pretrained=True).eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'models/resnet50.onnx',
    opset_version=12, do_constant_folding=False,
    input_names=['input'], output_names=['output'])
"

# MobileNetV3-Small
python -c "
import torch, torchvision
model = torchvision.models.mobilenet_v3_small(pretrained=True).eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'models/mobilenetv3_small.onnx',
    opset_version=12, do_constant_folding=False,
    input_names=['input'], output_names=['output'])
"

# EfficientDet-D0 (ONNX Model Zoo)
wget -P models/ https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/efficientdet/model/efficientdet-d0.onnx
```

### Group B — Transformers

```bash
pip install transformers torch

python -c "
from transformers import BertModel, BertTokenizer
import torch

for model_name, fname in [
    ('bert-base-uncased', 'bert_base'),
    ('bert-large-uncased', 'bert_large'),
    ('distilbert-base-uncased', 'distilbert_base'),
    ('roberta-base', 'roberta_base'),
    ('albert-base-v2', 'albert_base'),
]:
    print(f'Exporting {model_name}...')
    model = BertModel.from_pretrained(model_name).eval()
    dummy_ids = torch.ones(1, 128, dtype=torch.long)
    dummy_mask = torch.ones(1, 128, dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        f'models/{fname}.onnx',
        opset_version=14,
        do_constant_folding=False,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq'},
            'attention_mask': {0: 'batch', 1: 'seq'},
        }
    )
    print(f'  -> models/{fname}.onnx')
"

# Whisper
python -c "
import torch
from transformers import WhisperForConditionalGeneration

for variant, fname in [('openai/whisper-tiny', 'whisper_tiny'), ('openai/whisper-base', 'whisper_base')]:
    print(f'Exporting {variant}...')
    model = WhisperForConditionalGeneration.from_pretrained(variant).eval()
    encoder = model.model.encoder
    dummy_input = torch.randn(1, 80, 3000)
    torch.onnx.export(
        encoder,
        dummy_input,
        f'models/{fname}_encoder.onnx',
        opset_version=14,
        do_constant_folding=False,
        input_names=['input_features'],
        output_names=['last_hidden_state'],
    )
    print(f'  -> models/{fname}_encoder.onnx')
"

# MobileViT-Small
python -c "
import torch
from transformers import MobileViTModel

model = MobileViTModel.from_pretrained('apple/mobilevit-small').eval()
dummy = torch.randn(1, 3, 256, 256)
torch.onnx.export(model, dummy, 'models/mobilevit_small.onnx',
    opset_version=14, do_constant_folding=False,
    input_names=['pixel_values'], output_names=['last_hidden_state'])
"

# DeiT-Small
python -c "
import torch
from transformers import DeiTModel

model = DeiTModel.from_pretrained('facebook/deit-small-patch16-224').eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'models/deit_small.onnx',
    opset_version=14, do_constant_folding=False,
    input_names=['pixel_values'], output_names=['last_hidden_state'])
"
```

---

## Experiments

### Exp 01 — Full Pass Attribution Matrix (All 12 Models)

**The master table. Every row is a model. Every column is a pass. Every cell is the node delta.**

Run each pass in isolation on every model. Record the exact node count change.

```
Expected output:

Pass Attribution Matrix (Δ nodes, isolated runs)

Pass                          | YOLOv8n | YOLOv8s | ResNet | MobileV3 | BERT-base | BERT-large | DistilBERT | RoBERTa | Whisper-tiny | Whisper-base | MobileViT | DeiT
------------------------------|---------|---------|--------|----------|-----------|------------|------------|---------|--------------|--------------|-----------|-----
eliminate_dead_nodes          |    ?    |    ?    |   ?    |    ?     |     ?     |     ?      |     ?      |    ?    |      ?       |      ?       |     ?     |  ?
eliminate_identity_ops        |    ?    |    ?    |   ?    |    ?     |     ?     |     ?      |     ?      |    ?    |      ?       |      ?       |     ?     |  ?
eliminate_unused_initializers |    ?    |    ?    |   ?    |    ?     |     ?     |     ?      |     ?      |    ?    |      ?       |      ?       |     ?     |  ?
eliminate_duplicate_constants |    ?    |    ?    |   ?    |    ?     |     ?     |     ?      |     ?      |    ?    |      ?       |      ?       |     ?     |  ?
eliminate_redundant_transposes|    ?    |    ?    |   ?    |    ?     |     ?     |     ?      |     ?      |    ?    |      ?       |      ?       |     ?     |  ?
fold_constants                |    ?    |    ?    |   ?    |    ?     |     ?     |     ?      |     ?      |    ?    |      ?       |      ?       |     ?     |  ?
simplify_shape_chains         |    ?    |    ?    |   ?    |    ?     |     ?     |     ?      |     ?      |    ?    |      ?       |      ?       |     ?     |  ?
fuse_conv_batchnorm           |    ?    |    ?    |   ?    |    ?     |     0     |     0      |     0      |    0    |      0       |      0       |     ?     |  0
fuse_conv_relu                |    ?    |    ?    |   ?    |    ?     |     0     |     0      |     0      |    0    |      0       |      0       |     ?     |  0
fuse_matmul_add               |    ?    |    ?    |   ?    |    ?     |     ?     |     ?      |     ?      |    ?    |      ?       |      ?       |     ?     |  ?
cleanup_attention             |    ?    |    ?    |   0    |    0     |     ?     |     ?      |     ?      |    ?    |      ?       |      ?       |     ?     |  ?
TOTAL                         |    ?    |    ?    |   ?    |    ?     |     ?     |     ?      |     ?      |    ?    |      ?       |      ?       |     ?     |  ?
```

**Hard gates for Exp 01:**
- `eliminate_redundant_transposes` must show non-zero on at least one YOLO variant
- `fold_constants` must show non-zero on at least one Transformer model
- `fuse_matmul_add` must show non-zero on at least one Transformer model
- `cleanup_attention` must show non-zero on BERT-base

---

### Exp 02 — YOLOv8 Transpose Deep Dive

**Goal:** Prove that YOLOv8 exports have cancelling or composable Transpose pairs
and that M4 fires correctly.

```python
# What to measure:
# 1. Total Transpose nodes before optimization
# 2. Cancelling pairs (A→B where compose(A,B) = identity)
# 3. Composable pairs (A→B where compose(A,B) ≠ identity but single Transpose)
# 4. Surviving Transposes after M4

# Expected output:
YOLOv8n Transpose Analysis:
  Transpose nodes before:    XX
  Cancelling pairs found:    XX  → removed
  Composable pairs found:    XX  → merged to single
  Transposes remaining:      XX
  Node reduction from M4:    XX
```

---

### Exp 03 — BERT Full Pipeline Benchmark

**Goal:** End-to-end benchmark for BERT-base. This is the primary number for the README.

```
Model: models/bert_base.onnx
─────────────────────────────────────────
Nodes before:      ?
Nodes after:       ?  (?%)
Size before:       ? MB
Size after:        ? MB
Accuracy delta:    ?e+?? (✓/✗)
Time to optimize:  ?s

Pass breakdown:
  eliminate_redundant_transposes: -?
  fold_constants:                 -?
  simplify_shape_chains:          -?
  fuse_matmul_add:                -?
  cleanup_attention:              -?
```

**Hard gate:** BERT must show >5% node reduction. If not, cleanup_attention needs revision.

---

### Exp 04 — Attention Cleanup Deep Dive (BERT + Whisper)

**Goal:** Document exactly what cleanup_attention does inside attention blocks.
This is the differentiator pass. It needs forensic documentation.

```
Before cleanup_attention (BERT attention head, 1 layer):
  [diagram of nodes: Reshape → Transpose → MatMul → Div → Add → Softmax → Transpose → Reshape → MatMul]
  Redundant Reshape count: ?
  Redundant Transpose count: ?

After cleanup_attention:
  [simplified diagram]
  Nodes removed: ?
  Mathematical equivalence: verified (max_diff=?)

Whisper cross-attention variant:
  Different? Y/N
  Additional patterns found: ?
  Additional nodes removed: ?
```

---

### Exp 05 — Pass Order Sensitivity (Extended: Multi-Pass Interaction)

**Goal:** M9 showed order doesn't matter for models where only 1 pass fires.
Now test models where 3+ passes fire simultaneously (BERT, Whisper).

```
Models: BERT-base, Whisper-tiny (where 3+ passes fire)
Method: Run all permutations of {fold_constants, shape_chains, matmul_add, cleanup_attention}
        That's 4! = 24 permutations per model.
        Measure: final node count, accuracy (max_diff)

Hypothesis:
  fold_constants MUST precede simplify_shape_chains (shape chains only
  become foldable after constants are resolved)

Expected finding: DAG of pass dependencies
```

**This experiment, if it finds ordering matters, is publishable standalone.**

---

### Exp 06 — Tolerance Sweep (Extended to All New Models)

**Goal:** Verify perfect accuracy across all 12 new models, not just MobileNetV2 + EfficientNet.

```
Run 20 random seeds per model.
Record: min, max, mean, p99 of max_diff.

Expected:
  Vision models (ResNet, YOLO): all_zero=True (fusion is provably lossless)
  Transformer models (BERT, Whisper): all_zero=True or epsilon < 1e-5

Any model showing max_diff > 1e-5 is a bug to fix before M10 closes.
```

---

### Exp 07 — Model Size vs. Node Count Divergence

**New in M10.** Tests a hypothesis nobody has verified: does node count reduction
translate proportionally to model size reduction?

```
M9 showed: EfficientNet lost 49 nodes (-17%) but SIZE INCREASED (+0.1MB)
Why? BN parameters folded into Conv weights (redistributed, not eliminated)

Hypothesis: Node reduction and size reduction are NOT correlated for fusion passes.
            They ARE correlated for elimination passes.

Experiment:
  For each pass, measure both Δ nodes AND Δ size independently
  Build correlation matrix: which passes reduce nodes? which reduce size? both? neither?

Expected findings:
  eliminate_* passes: reduce nodes AND size (unused weights removed)
  fuse_* passes: reduce nodes, MAY increase size (parameters absorbed)
  fold_constants: reduces nodes, reduces size (eliminates subgraph entirely)
```

**This is paper-section material.** The distinction between node reduction and size reduction
has not been characterized in the literature.

---

### Exp 08 — Graph Inspector (All 12 Models)

Run exp_08_graph_inspector.py on every new model before optimization.
Build the complete op-frequency baseline table.

```
Expected output (12-model op frequency table):

Model          | Nodes | Conv | BN | Relu | Clip | MatMul | Transpose | Reshape | Softmax | LayerNorm
---------------|-------|------|----|------|------|--------|-----------|---------|---------|----------
YOLOv8n        |   ?   |   ?  |  ? |   ?  |   ?  |    ?   |     ?     |    ?    |    ?    |    ?
YOLOv8s        |   ?   |   ?  |  ? |   ?  |   ?  |    ?   |     ?     |    ?    |    ?    |    ?
ResNet-50      |   ?   |   ?  |  ? |   ?  |   ?  |    ?   |     ?     |    ?    |    ?    |    ?
MobileNetV3    |   ?   |   ?  |  ? |   ?  |   ?  |    ?   |     ?     |    ?    |    ?    |    ?
BERT-base      |   ?   |   0  |  0 |   0  |   0  |    ?   |     ?     |    ?    |    ?    |    ?
BERT-large     |   ?   |   0  |  0 |   0  |   0  |    ?   |     ?     |    ?    |    ?    |    ?
DistilBERT     |   ?   |   0  |  0 |   0  |   0  |    ?   |     ?     |    ?    |    ?    |    ?
RoBERTa        |   ?   |   0  |  0 |   0  |   0  |    ?   |     ?     |    ?    |    ?    |    ?
Whisper-tiny   |   ?   |   ?  |  ? |   ?  |   ?  |    ?   |     ?     |    ?    |    ?    |    ?
Whisper-base   |   ?   |   ?  |  ? |   ?  |   ?  |    ?   |     ?     |    ?    |    ?    |    ?
MobileViT      |   ?   |   ?  |  ? |   ?  |   ?  |    ?   |     ?     |    ?    |    ?    |    ?
DeiT-Small     |   ?   |   0  |  0 |   0  |   0  |    ?   |     ?     |    ?    |    ?    |    ?
```

---

### Exp 09 — Cross-BERT-Family Consistency

**New in M10.** BERT, DistilBERT, RoBERTa, ALBERT all have similar attention structure
but different export signatures. Does cleanup_attention handle all variants?

```
For each BERT-family model:
  1. Run graph inspector → document attention pattern (op sequence in attention block)
  2. Run cleanup_attention in isolation → measure Δ nodes
  3. Compare patterns: are they the same subgraph? different? partially overlapping?

Expected finding:
  BERT-base and RoBERTa: identical attention export → same cleanup
  DistilBERT: 6 layers vs 12, but same per-layer pattern
  ALBERT: parameter sharing → unique graph structure → may need new pattern
```

**This directly feeds cleanup_attention generalization and Experiment Delta (architecture fingerprinting).**

---

### Exp 10 — Latency Benchmark (First Real Performance Numbers)

**New in M10.** Node count is a proxy metric. Latency is the real metric.

```
Setup:
  Runtime: ONNX Runtime on Mac M3 (CPU) — you have this
  Method: median of 50 inference runs, warm-up 5 runs
  Input: fixed batch size 1

Models to benchmark (latency before vs. after):
  YOLOv8n       (transpose reduction expected to help)
  EfficientNet  (BN fusion expected to help)
  BERT-base     (full pipeline — most important number)
  Whisper-tiny  (encoder only)

Report format:
  Model          | Before (ms) | After (ms) | Speedup | Node Δ
  ---------------|-------------|------------|---------|--------
  YOLOv8n        |     ?       |     ?      |   ?x    |   ?
  EfficientNet   |     ?       |     ?      |   ?x    |   ?
  BERT-base      |     ?       |     ?      |   ?x    |   ?
  Whisper-tiny   |     ?       |     ?      |   ?x    |   ?
```

**Hard gate:** At least one model must show measurable latency improvement (>2%).
If zero latency improvement across all models, the optimizer's practical value
needs to be reframed around conversion success rate, not inference speed.

---

### Exp 11 — cleanup_attention Revision (If Exp 03 Shows <5% BERT Reduction)

**Contingency experiment.** If BERT shows less than 5% node reduction, cleanup_attention
needs forensic debugging.

```
Process:
  1. Run graph inspector on BERT-base → document every Reshape, Transpose in attention blocks
  2. Run cleanup_attention in isolation → log every pattern it tries to match
  3. Log every pattern match FAILURE (pattern found but condition not met)
  4. Identify the gap between what the pass expects and what the export produces
  5. Revise pattern-matching logic
  6. Rerun → verify improvement
```

This is not a nice-to-have. If BERT doesn't show meaningful reduction, the most
important pass in the pipeline is broken and M10 cannot close.

---

## New Pass Candidates (Implement in M10 If Time Allows)

### Pass 12 — decompose_layernorm (TFLite Prerequisite)

TFLite does not support LayerNorm natively. BERT and Whisper use LayerNorm.
Without this pass, the TFLite conversion step (Phase 2) will fail on both models.

```python
# Pattern to detect:
# LayerNorm → decompose to: Sub → Pow(2) → ReduceMean → Add(eps) → Sqrt → Div → Mul(gamma) → Add(beta)

# Why now: every Transformer model uses LayerNorm
# TFLite target profile will call this pass automatically
```

### Pass 13 — freeze_dynamic_shapes (TFLite Prerequisite)

TFLite requires static shapes. HuggingFace exports often have dynamic batch/seq dimensions.

```python
# Pattern: dynamic axes in model inputs (batch=None, seq=None)
# Fix: freeze to concrete values (batch=1, seq=128 for BERT)
# Must document which axes were frozen for user transparency
```

### Pass 14 — einsum_to_matmul (TFLite Prerequisite)

TFLite does not handle Einsum. Several Transformer exports use Einsum for attention.

```python
# Pattern: Einsum with specific subscript strings
# Known safe conversions:
#   'bhqd,bhkd->bhqk'  → batched MatMul + Transpose
#   'bhqk,bhvd->bhqd'  → batched MatMul + Transpose
```

---

## Tests to Write

For every new model, add an integration test:

```
tests/
├── test_yolov8n.py          ← confirm transposes fire
├── test_yolov8s.py
├── test_resnet50.py
├── test_mobilenetv3.py
├── test_bert_base.py        ← THE critical test
├── test_bert_large.py
├── test_distilbert.py
├── test_roberta.py
├── test_whisper_tiny.py
├── test_whisper_base.py
├── test_mobilevit.py
└── test_deit_small.py
```

Each test must:
1. Run optimizer on the model
2. Verify accuracy (max_diff < 1e-5)
3. Assert node count decreased by expected amount
4. Assert key passes fired (non-zero reduction for passes expected to fire)

---

## Hard Gates (M10 Does Not Close Until All Pass)

| Gate | Threshold | Reason |
|------|-----------|--------|
| eliminate_redundant_transposes fires on real model | ≥1 YOLO model, ≥10 nodes removed | Proves M4 is real |
| fold_constants fires on real Transformer | ≥1 BERT-family model, non-zero | Proves M6 handles Transformer exports |
| fuse_matmul_add fires on real Transformer | ≥1 BERT-family model, non-zero | Proves M8 correct |
| cleanup_attention fires on BERT-base | ≥5% node reduction on BERT | Core differentiator must work |
| Accuracy verified across all 12 models | max_diff < 1e-5 on every model | Non-negotiable correctness |
| Latency improvement on at least 1 model | >2% speedup on ORT CPU | Practical value |
| Attribution matrix complete | No TBD cells | Research foundation |
| All 12 model tests pass | 12/12 green | CI gate |
| README benchmark table complete | All rows filled | Public credibility |

---

## README Benchmark Table (Target State After M10)

```
| Model            | Nodes Before | Nodes After | Reduction | Size Δ  | Latency Δ | max_diff  |
|------------------|-------------|-------------|-----------|---------|-----------|-----------|
| MobileNetV2      | 105         | 105         | 0%        | 0%      | -         | 0.00e+00  |
| EfficientNet-B0  | 288         | 239         | 17%       | +0.1MB  | ?         | 0.00e+00  |
| YOLOv8n          | TBD         | TBD         | TBD       | TBD     | TBD       | TBD       |
| YOLOv8s          | TBD         | TBD         | TBD       | TBD     | TBD       | TBD       |
| ResNet-50        | TBD         | TBD         | TBD       | TBD     | TBD       | TBD       |
| MobileNetV3-S    | TBD         | TBD         | TBD       | TBD     | TBD       | TBD       |
| BERT-base        | TBD         | TBD         | TBD       | TBD     | TBD       | TBD       |
| BERT-large       | TBD         | TBD         | TBD       | TBD     | TBD       | TBD       |
| DistilBERT       | TBD         | TBD         | TBD       | TBD     | TBD       | TBD       |
| RoBERTa-base     | TBD         | TBD         | TBD       | TBD     | TBD       | TBD       |
| Whisper-tiny     | TBD         | TBD         | TBD       | TBD     | TBD       | TBD       |
| Whisper-base     | TBD         | TBD         | TBD       | TBD     | TBD       | TBD       |
| MobileViT-S      | TBD         | TBD         | TBD       | TBD     | TBD       | TBD       |
| DeiT-Small       | TBD         | TBD         | TBD       | TBD     | TBD       | TBD       |
```

---

## Build Order

### Week 1 — Vision Models
1. Download YOLOv8n, YOLOv8s via ultralytics
2. Run Exp 08 (graph inspector) on both
3. Run full optimizer pipeline on YOLOv8n
4. Run Exp 02 (Transpose deep dive) on YOLOv8n
5. Download ResNet-50, MobileNetV3 via torchvision
6. Run graph inspector + full pipeline on both
7. Write tests: test_yolov8n.py, test_yolov8s.py, test_resnet50.py, test_mobilenetv3.py
8. Fill vision rows in attribution matrix

### Week 2 — BERT Family
1. Export BERT-base, BERT-large, DistilBERT, RoBERTa, ALBERT via transformers
2. Run Exp 08 on all five — document attention op patterns
3. Run full pipeline on BERT-base (Exp 03)
4. Run Exp 04 (attention cleanup deep dive) on BERT-base
5. If BERT reduction < 5% → run Exp 11 (cleanup_attention revision)
6. Run cross-BERT consistency (Exp 09)
7. Write all 5 BERT-family tests

### Week 3 — Whisper + Hybrids
1. Export Whisper-tiny, Whisper-base encoder via transformers
2. Compare attention patterns BERT vs. Whisper (Exp 04 extension)
3. Export MobileViT-Small, DeiT-Small
4. Run graph inspector + full pipeline
5. Write tests for all four models

### Week 4 — Experiments + New Passes
1. Run Exp 05 (pass order, multi-pass interaction) on BERT + Whisper
2. Run Exp 06 (tolerance sweep) on all 12 new models
3. Run Exp 07 (size vs. node count divergence)
4. Run Exp 10 (latency benchmark: YOLO + EfficientNet + BERT + Whisper)
5. Implement Pass 12 (decompose_layernorm) if BERT tests reveal TFLite need
6. Implement Pass 13 (freeze_dynamic_shapes) if Whisper export has dynamic axes
7. Fill attribution matrix completely
8. Update README benchmark table — all TBDs resolved

---

## Key Insights to Capture

After running all experiments, document:

1. **Which model family benefits most per pass** — this becomes the auto-selection logic for Phase 3 CLI
2. **Whether node reduction and size reduction are correlated** — Exp 07 answers this (likely: no for fusion passes)
3. **Whether cleanup_attention generalizes across BERT variants** — Exp 09 answers this
4. **Whether pass order matters when 3+ passes fire** — Exp 05 answers this (may be first publishable finding)
5. **First latency numbers** — Exp 10 — this is what goes in the Phase 2 pitch

---

## What Comes After M10

Once M10 closes with all 14 models tested and all experiments complete:

**M11** — TFLite Conversion Pipeline
- Wrap onnxslim + TFLite converter into single pipeline
- Use decompose_layernorm and freeze_dynamic_shapes from M10
- First end-to-end conversion: ONNX → onnxslim → TFLite
- Benchmark: does onnxslim reduce TFLite conversion failure rate?

**M12** — CoreML Conversion Pipeline
- Same wrapper for CoreML Tools
- Use M3 Mac for native CoreML benchmarks
- First cross-platform number: same model, TFLite vs. CoreML latency

**Research Agenda**
- M10 attribution matrix → Experiment Alpha (accuracy taxonomy)
- M10 pass order results → Experiment Beta (phase ordering paper)
- M10 latency numbers → Experiment Gamma (quantization preprocessing)

M10 is the foundation everything else stands on.
Every number in the attribution matrix is a citation.
Every hard gate closed is a claim you can defend.
