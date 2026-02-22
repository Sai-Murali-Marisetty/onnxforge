# Micro-plan M11 — Crack the Transformer Problem

## What M10 Actually Told Us

M10 closed with the most important finding so far:

```
Vision models:    17–32% node reduction   ✅ passes work
Transformer models:  0% node reduction    ❌ passes don't touch them
```

This is not a failure. This is a diagnosis. Every BERT, RoBERTa, DistilBERT,
Whisper export came out of M10 identical to what went in. That means the
transformer graph has structure our passes don't understand yet.

M10 also surfaced the exact reason: **MatMul+Add fusion requires rank-2 tensors.
Transformers use 3D MatMuls.** That's one known gap. The question M11 answers
is: how many more gaps are there, what are they, and can we close them?

M11 is the milestone that either makes onnxforge the best transformer optimizer
available — or reveals that the problem is harder than expected and we need
a different approach. Either outcome is valuable. Unknown outcomes are not.

---

## Status: 🔴 NOT STARTED

---

## The Central Hypothesis

```
Hypothesis: HuggingFace transformer exports contain 5+ categories of redundant
or suboptimal graph structure that our current passes miss entirely, each of which
is fixable with a targeted new pass.

If true: BERT goes from 0% → 15–25% node reduction.
If false: transformer graphs are already optimal from HF's exporter — that itself
is a publishable finding (no tool can optimize them further without accuracy loss).

Either way, we know after M11.
```

---

## Step 0 — The Forensic Audit (Do This Before Writing Any Code)

Before implementing a single new pass, dissect a BERT-base export with surgical
precision. Run this audit on `bert_base.onnx`. Every observation feeds directly
into which new passes to build.

### Audit Script: `tools/graph_forensics.py`

```python
"""
Deep audit of a transformer ONNX graph.
Produces a structured report of every optimization opportunity.
"""

def audit_transformer(model_path):
    model = onnx.load(model_path)
    graph = model.graph

    report = {
        # 1. Shape chain analysis
        "shape_gather_unsqueeze_concat": find_shape_chains(graph),

        # 2. Consecutive ops of same type
        "consecutive_reshape": find_consecutive(graph, "Reshape"),
        "consecutive_transpose": find_consecutive(graph, "Transpose"),
        "consecutive_cast": find_consecutive(graph, "Cast"),

        # 3. Identity-equivalent ops
        "noop_transpose": find_identity_transposes(graph),   # perm=[0,1,2,3]
        "noop_reshape": find_identity_reshapes(graph),       # output shape = input shape

        # 4. 3D MatMul patterns
        "matmul_3d": find_3d_matmuls(graph),
        "matmul_add_3d": find_3d_matmul_add(graph),         # The known gap from M10

        # 5. Attention block structure
        "attention_blocks": find_attention_blocks(graph),    # Q/K/V projection pattern
        "attention_reshapes": count_attention_reshapes(graph),

        # 6. Constant subgraphs not yet folded
        "unfused_constants": find_constant_subgraphs(graph),

        # 7. Cast chains
        "cast_chains": find_cast_chains(graph),              # Cast → Cast → Cast
        "redundant_casts": find_redundant_casts(graph),      # Cast to same dtype

        # 8. LayerNorm pattern
        "layernorm_nodes": find_layernorm_pattern(graph),

        # 9. Div by sqrt(d) in attention
        "attention_scale_pattern": find_attention_scale(graph),

        # 10. Expand + Add broadcast patterns
        "expand_add_pairs": find_expand_add(graph),
    }

    return report
```

**Run this on: BERT-base, DistilBERT, RoBERTa, Whisper-tiny encoder.**
Document every non-zero finding. Each non-zero entry is a pass candidate.

---

## New Models to Download

### Group D — LLMs (The Differentiator)

| Model | Source | Why |
|-------|--------|-----|
| GPT-2 Small (124M) | HuggingFace | Decoder-only — different attention than BERT |
| TinyLlama-1.1B | HuggingFace (quantized) | Modern LLM at testable size |
| Phi-1.5 (1.3B) | HuggingFace | Microsoft's efficient LLM — strong on-device story |
| Gemma-2B (if RAM allows) | HuggingFace | Google's LLM — on-device target |

### Group E — Vision Transformers (Hybrid Pass Test)

| Model | Source | Why |
|-------|--------|-----|
| YOLOv8n | ultralytics | Still deferred from M10 — transpose pass proof |
| DeiT-Small | HuggingFace | Pure vision transformer, no Conv |
| MobileViT-Small | HuggingFace | Hybrid CNN+Transformer — tests both families |
| ViT-Base-Patch16 | HuggingFace | Canonical vision transformer baseline |

### Group F — Audio

| Model | Source | Why |
|-------|--------|-----|
| Whisper-base encoder | HuggingFace | Scale up from tiny |
| Wav2Vec2-Base | HuggingFace | CNN + transformer encoder — hybrid |

### Download Scripts

```bash
# GPT-2
python -c "
import torch
from transformers import GPT2Model
model = GPT2Model.from_pretrained('gpt2').eval()
dummy_ids = torch.ones(1, 128, dtype=torch.long)
torch.onnx.export(model, dummy_ids, 'models/gpt2_small.onnx',
    opset_version=14, do_constant_folding=False,
    input_names=['input_ids'], output_names=['last_hidden_state'])
print('-> gpt2_small.onnx')
"

# TinyLlama
python -c "
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    torch_dtype=torch.float32).eval()
dummy = torch.ones(1, 32, dtype=torch.long)
torch.onnx.export(model, dummy, 'models/tinyllama.onnx',
    opset_version=17, do_constant_folding=False,
    input_names=['input_ids'], output_names=['logits'])
print('-> tinyllama.onnx')
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

# ViT-Base
python -c "
import torch
from transformers import ViTModel
model = ViTModel.from_pretrained('google/vit-base-patch16-224').eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'models/vit_base.onnx',
    opset_version=14, do_constant_folding=False,
    input_names=['pixel_values'], output_names=['last_hidden_state'])
"

# YOLOv8n (finally)
pip install ultralytics
python -c "
from ultralytics import YOLO
YOLO('yolov8n.pt').export(format='onnx', opset=12, dynamic=False)
YOLO('yolov8s.pt').export(format='onnx', opset=12, dynamic=False)
"

# Wav2Vec2
python -c "
import torch
from transformers import Wav2Vec2Model
model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base').eval()
dummy = torch.randn(1, 16000)
torch.onnx.export(model, dummy, 'models/wav2vec2_base.onnx',
    opset_version=14, do_constant_folding=False,
    input_names=['input_values'], output_names=['last_hidden_state'])
"

# Whisper-base encoder
python -c "
import torch
from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base').eval()
encoder = model.model.encoder
dummy = torch.randn(1, 80, 3000)
torch.onnx.export(encoder, dummy, 'models/whisper_base_encoder.onnx',
    opset_version=14, do_constant_folding=False,
    input_names=['input_features'], output_names=['last_hidden_state'])
"
```

---

## New Passes to Implement

### Pass 12 — fuse_matmul_add_3d (Fix the Known Gap)

**The exact gap M10 revealed.** Our current fuse_matmul_add requires rank-2 for Gemm.
Transformers use (batch, seq, hidden) — rank 3. This pass handles 3D specifically.

```python
# Pattern:
#   MatMul (3D: B×S×H @ H×H) → Add (bias: H)
# Options:
#   Option A: Reshape to 2D → Gemm → Reshape back  (standard trick)
#   Option B: Keep as MatMul but fold bias into initializer  (simpler, always valid)
#   Option C: Use new ONNX opset 13+ Gemm that handles batching
#
# Recommended: Option B first (no reshape overhead), then benchmark A vs B

# Expected impact: BERT has 144 MatMul+Add sequences (12 layers × 12 heads)
# If each fuses → -144 nodes on BERT-base alone
```

### Pass 13 — eliminate_noop_transposes

**Distinct from eliminate_redundant_transposes (which removes pairs).
This removes single transposes that are mathematical identity.**

```python
# Pattern:
#   Transpose(x, perm=[0, 1, 2, 3])  → output is identical to input
#   Transpose(x, perm=[0, 1, 2])     → output is identical to input
#   Any Transpose where perm is the identity permutation

# Why it appears:
#   HF exporter sometimes inserts these when converting attention heads
#   between NCHW and NHWC mentally but the actual perm ends up as identity

# Expected impact: Run audit first — may be 0, may be significant
```

### Pass 14 — eliminate_noop_reshapes

**Reshapes that don't change the shape.**

```python
# Pattern:
#   input shape: [1, 128, 768]
#   Reshape → [1, 128, 768]   ← does nothing
#   Replace: remove the Reshape node, connect input directly to output consumers

# Why it appears:
#   HF attention blocks often have "safety reshapes" that are artifacts of
#   the original Python code being overly explicit about tensor shapes

# Expected impact: Audit first. BERT attention blocks are candidates.
```

### Pass 15 — fuse_attention_qkv (The Differentiator, Hard Version)

**This is the pass that would make onnxforge unique. Ambitious but defined.**

Standard attention in a HuggingFace BERT export looks like:

```
input
  ├── MatMul(Wq) → Reshape → Transpose   \
  ├── MatMul(Wk) → Reshape → Transpose    → Attention computation
  └── MatMul(Wv) → Reshape → Transpose   /
```

Three separate MatMuls + 6 Reshapes + 3 Transposes for every layer.

Fused version (what TensorRT and CoreML prefer):
```
input → single QKV MatMul(W_qkv concatenated) → split → attention
```

This is not lossless in the general sense — it changes graph structure.
But it is **mathematically equivalent** and **verifiable**.

```python
# Implementation approach:
# 1. Pattern match: 3 MatMuls with same input, different weight initializers
# 2. Verify weight shapes are compatible for concatenation
# 3. Concatenate Wq, Wk, Wv into W_qkv
# 4. Replace 3 MatMuls with 1 MatMul + 1 Split
# 5. Verify output (max_diff must be 0.0)

# Expected impact:
#   BERT-base: 12 layers × 3 MatMuls = 36 MatMuls → 12 MatMuls + 12 Splits
#   Net: -24 MatMul nodes (but +12 Split nodes)
#   Real value: single larger GEMM is faster than three small GEMMs on NPU

# Risk: HIGH. Weight concatenation must be exact. Verify hard.
# Build this last, after Passes 12-14 are proven.
```

### Pass 16 — decompose_layernorm (TFLite Prerequisite)

```python
# Pattern: LayerNorm op (opset 17+) or custom LayerNorm subgraph
# Replace with: Sub → Pow(2) → ReduceMean → Add(eps) → Sqrt → Div → Mul(gamma) → Add(beta)
# Why: TFLite has no LayerNorm op. Without this, TFLite conversion fails on every transformer.
# This is not optional — it's required for Phase 2 (conversion pipeline).

# Also: Some exports already decompose this. Audit first to see if BERT uses the op or
# the subgraph form. If subgraph form: this pass is a no-op and TFLite will handle it.
```

### Pass 17 — fold_cast_chains

```python
# Pattern:
#   Cast(x, to=FLOAT) → Cast(y, to=FLOAT16) → Cast(z, to=FLOAT)
#   = net no-op if first and last dtype are the same

# Pattern 2:
#   Cast(x, to=FLOAT) → Cast(y, to=FLOAT)
#   = remove second Cast (already correct dtype)

# Why it appears:
#   Mixed-precision exports, quantization traces, HF optimum exports
#   TinyLlama and modern LLMs are especially prone to this
```

### Pass 18 — eliminate_expand_identity

```python
# Pattern:
#   Expand(x, shape) where shape equals x.shape
#   = Expand does nothing, replace with Identity, then eliminate

# Why it appears:
#   Attention mask broadcasting in BERT/RoBERTa
#   Positional embedding broadcasting in GPT-2/Whisper
```

---

## Experiments

### Exp 11 — Transformer Graph Forensics (Audit Before Building)

**Run first. Results determine which passes to build.**

```
For each model in [BERT-base, DistilBERT, RoBERTa, GPT-2, Whisper-tiny, TinyLlama]:
  Run graph_forensics.py
  Record findings in the Forensics Table

Forensics Table:
  Finding                      | BERT | DistilBERT | RoBERTa | GPT-2 | Whisper | TinyLlama
  -----------------------------|------|------------|---------|-------|---------|----------
  3D MatMul+Add pairs          |  ?   |     ?      |    ?    |   ?   |    ?    |    ?
  Noop transposes (identity)   |  ?   |     ?      |    ?    |   ?   |    ?    |    ?
  Noop reshapes                |  ?   |     ?      |    ?    |   ?   |    ?    |    ?
  QKV triple-MatMul pattern    |  ?   |     ?      |    ?    |   ?   |    ?    |    ?
  LayerNorm op (not decomposed)|  ?   |     ?      |    ?    |   ?   |    ?    |    ?
  Cast chains                  |  ?   |     ?      |    ?    |   ?   |    ?    |    ?
  Expand identity patterns     |  ?   |     ?      |    ?    |   ?   |    ?    |    ?
  Unfused constant subgraphs   |  ?   |     ?      |    ?    |   ?   |    ?    |    ?

This table determines which new passes have non-zero impact before we write them.
DO NOT implement a pass that shows 0 in the forensics table.
```

---

### Exp 12 — 3D MatMul+Add Fusion Impact (Pass 12 Proof)

**The known fix from M10. This is the first thing to implement and measure.**

```
Baseline (M10 result):
  BERT-base: 1453 nodes → 1453 nodes (0% reduction)

After Pass 12:
  Expected: BERT-base → ? nodes

Measure:
  - How many MatMul+Add pairs exist in BERT (run forensics first)
  - How many fuse successfully (may not be all — bias shape matters)
  - Node count before/after
  - Accuracy: max_diff must be 0.0
  - Latency: does fusing 3D MatMul+Add actually help ORT?

Success condition:
  BERT-base shows >5% node reduction from this pass alone.
  If not, document exactly why (shape mismatch? opset constraint? other reason?)
```

---

### Exp 13 — Noop Elimination Audit (Passes 13 + 14)

```
Run noop detection on all transformer models:
  - Count identity transposes per model
  - Count identity reshapes per model

If either shows >10 noops on BERT → implement the pass and measure.
If both show <5 noops on every model → skip the passes (not worth the code).

This is how you decide what to build. Don't build blindly.
```

---

### Exp 14 — QKV Fusion Feasibility Study (Pass 15 Pre-Study)

**Before building the hardest pass, prove it's worth building.**

```
Step 1: Detect QKV pattern
  - Write a detector (not a pass) that finds triple-MatMul QKV blocks
  - Count: how many per layer? per model?
  - Verify: are Wq, Wk, Wv always the same shape?

Step 2: Manual prototype on one attention block
  - Manually concatenate one set of Wq, Wk, Wv
  - Run both versions through ORT
  - Confirm max_diff = 0.0

Step 3: Measure theoretical speedup
  - 3 MatMuls (H×H each) vs 1 MatMul (3H×H)
  - On ORT CPU: does larger single GEMM beat 3 small ones?
  - Benchmark: latency before/after

If max_diff > 0 at any point in Step 2 → do not implement Pass 15.
QKV fusion requires perfect numerical precision or it's wrong by definition.
```

---

### Exp 15 — YOLOv8n Transpose Deep Dive (Deferred from M10)

**Still the missing proof for eliminate_redundant_transposes.**

```
Finally run YOLOv8n through the full pipeline.

Expected:
  - 150-300+ transpose nodes
  - Significant cancelling pairs (NCHW→NHWC→NCHW pattern)
  - 10–50 node reduction from transpose elimination alone

Measure:
  - Transpose count before
  - Cancelling pairs found
  - Composable pairs found
  - Transposes after
  - Node reduction (absolute + %)
  - Accuracy: max_diff
  - Latency before/after

This finally closes the gap in the attribution table for eliminate_redundant_transposes.
```

---

### Exp 16 — Vision Transformer Hybrid Test (DeiT + MobileViT + ViT)

**Tests both pass families on hybrid models simultaneously.**

```
For each of [DeiT-Small, MobileViT-Small, ViT-Base]:
  Run graph inspector
  Run full pipeline
  Record: which passes fire? node reduction? accuracy?

Hypothesis:
  MobileViT: Conv+BN fusion fires (CNN component) + new transformer passes fire
  DeiT: Only transformer passes fire (no Conv)
  ViT-Base: Only transformer passes fire (no Conv)

Key question: Does DeiT/ViT show >0% reduction after new transformer passes?
If DeiT shows reduction but pure-BERT doesn't → the difference tells you something
about the attention implementation (DeiT uses more explicit reshape/transpose patterns)
```

---

### Exp 17 — GPT-2 vs BERT Graph Divergence

**First decoder-only transformer test.**

```
GPT-2 is architecturally different from BERT:
  - Unidirectional attention (causal mask)
  - No [CLS] token pooling
  - Different positional encoding (learned, not sinusoidal)
  - Decoder-only (no encoder-decoder cross-attention)

Run forensics on GPT-2.
Compare forensics table to BERT.
Key questions:
  1. Does GPT-2 have more/fewer noop ops than BERT?
  2. Do our new transformer passes fire on GPT-2?
  3. Are the attention patterns the same or different?

This feeds directly into architecture fingerprinting (Experiment Epsilon).
```

---

### Exp 18 — TinyLlama: First Modern LLM Test

**This is the headline experiment for M11.**

```
TinyLlama-1.1B is:
  - Small enough to export (1.1B params, ~4GB in float32)
  - Modern LLM architecture (Llama-style: GQA, RoPE, SwiGLU)
  - Used in real on-device deployment (Qualcomm, MediaTek, Apple)
  - The model class ZETIC is targeting

Run full pipeline.
Run forensics.
Record: which passes fire? What patterns does LLaMA architecture introduce?

New patterns to look for:
  - RoPE (Rotary Position Encoding) — unique subgraph, may be foldable
  - GQA (Grouped Query Attention) — different QKV shape than BERT
  - SwiGLU activation — not Relu or GELU, different pattern

This is the experiment that might show onnxforge's biggest differentiator:
optimizing LLM-style architectures that nothing else handles.
```

---

### Exp 19 — Pass Attribution Matrix (All 16 Models)

**Updated master table including all new models and new passes.**

```
Pass Attribution Matrix (M11 — Δ nodes, isolated runs)

Pass                          | BERT | DistilBERT | RoBERTa | GPT-2 | TinyLlama | Whisper-tiny | Whisper-base | YOLOv8n | YOLOv8s | DeiT | MobileViT | ViT-Base | Wav2Vec2
------------------------------|------|------------|---------|-------|-----------|--------------|--------------|---------|---------|------|-----------|----------|----------
[existing passes from M10]... |  0   |     0      |    0    |   ?   |     ?     |      0       |      ?       |    ?    |    ?    |  ?   |     ?     |    ?     |    ?
fuse_matmul_add_3d (new)      |  ?   |     ?      |    ?    |   ?   |     ?     |      ?       |      ?       |    0    |    0    |  ?   |     ?     |    ?     |    ?
eliminate_noop_transposes (new)|  ?  |     ?      |    ?    |   ?   |     ?     |      ?       |      ?       |    ?    |    ?    |  ?   |     ?     |    ?     |    ?
eliminate_noop_reshapes (new) |  ?   |     ?      |    ?    |   ?   |     ?     |      ?       |      ?       |    0    |    0    |  ?   |     ?     |    ?     |    ?
fuse_attention_qkv (new)      |  ?   |     ?      |    ?    |   ?   |     ?     |      ?       |      ?       |    0    |    0    |  ?   |     ?     |    ?     |    ?
fold_cast_chains (new)        |  ?   |     ?      |    ?    |   ?   |     ?     |      ?       |      ?       |    0    |    0    |  ?   |     ?     |    ?     |    ?
decompose_layernorm (new)     |  ?   |     ?      |    ?    |   ?   |     ?     |      ?       |      ?       |    0    |    0    |  ?   |     ?     |    ?     |    ?
TOTAL (M11 pipeline)          |  ?   |     ?      |    ?    |   ?   |     ?     |      ?       |      ?       |    ?    |    ?    |  ?   |     ?     |    ?     |    ?
```

---

### Exp 20 — Latency Benchmark Extended (All New Models)

```
Benchmark ORT CPU latency: before vs after M11 full pipeline.
Add to M10 latency table.

Focus especially on transformer models — M10 showed DistilBERT improved 2.6%
WITHOUT any transformer-specific passes. With new passes, target: >5%.

Model          | Before (ms) | After (ms) | Speedup | Passes that fired
---------------|-------------|------------|---------|------------------
BERT-base      |     ?       |     ?      |   ?x    |  ?
GPT-2-small    |     ?       |     ?      |   ?x    |  ?
TinyLlama      |     ?       |     ?      |   ?x    |  ?
DeiT-Small     |     ?       |     ?      |   ?x    |  ?
MobileViT-S    |     ?       |     ?      |   ?x    |  ?
YOLOv8n        |     ?       |     ?      |   ?x    |  ?
Wav2Vec2       |     ?       |     ?      |   ?x    |  ?
Whisper-base   |     ?       |     ?      |   ?x    |  ?
```

---

### Exp 21 — The Publishable One: Node Reduction vs. Latency Correlation

**New. This is Experiment Gamma's foundation.**

```
Question: Does node count reduction predict latency improvement?

M10 showed a strange result:
  - DistilBERT: 0% node reduction → 2.6% latency improvement
  - EfficientNet: 17% node reduction → only 1.3% latency improvement

Node reduction and latency are NOT the same thing.
This needs to be characterized formally.

Experiment:
  Collect (node_reduction%, latency_improvement%) for every model after M11
  Fit a linear model: does correlation exist?
  Separate by model family: vision vs. transformer vs. hybrid
  Separate by pass type: elimination vs. fusion

Hypothesis:
  Fusion passes improve latency more than elimination passes
  (fusing MatMul+Add removes a memory round-trip, not just a node count)
  Elimination passes reduce model size more than latency

If confirmed: this is a paper-section claim nobody has measured.
The title writes itself: "Node Count Reduction Is a Poor Proxy for Latency Improvement
in ONNX Graph Optimization"
```

---

### Exp 22 — Architecture Fingerprinting Prototype (Experiment Epsilon Seed)

**First pass at the fingerprinting idea.**

```
For each model, extract:
  - Op type histogram (as vector)
  - MatMul/Conv ratio
  - Reshape count normalized by total nodes
  - Transpose count normalized by total nodes
  - Presence of: Softmax, LayerNorm, GELU, SiLU, RoPE subgraph

Cluster the models. Do they naturally separate into:
  Vision (Conv-heavy), Transformer (MatMul-heavy), Hybrid, LLM?

If yes: we have the basis for automatic pass selection.
  Vision graph → apply BN fusion, transpose elimination
  Transformer graph → apply 3D MatMul fusion, attention cleanup
  LLM graph → apply RoPE folding, QKV fusion, cast chain elimination

Even a simple rule-based classifier here is a contribution.
The feature that makes onnxforge's CLI smart:
  onnxforge optimize model.onnx  ← auto-detects architecture, applies right passes
```

---

## Accuracy Taxonomy — First Formal Draft

**M11 should produce the first written version of the accuracy taxonomy.**
This is Experiment Alpha from the research roadmap.

```
Class 0 — Provably Lossless (write proofs)
  eliminate_dead_nodes        Proof: removing unreachable nodes doesn't change output
  eliminate_identity_ops      Proof: f(x) = x, composition unchanged
  eliminate_unused_init.      Proof: unused weights never affect any output
  eliminate_duplicate_const.  Proof: same tensor value, same result
  eliminate_noop_transposes   Proof: identity permutation is identity function
  eliminate_noop_reshapes     Proof: same shape in and out, no reordering
  fuse_conv_batchnorm         Proof: BN params folded by algebraic substitution

Class 1 — Float32-Bounded (derive epsilon, verify empirically)
  fold_constants              Epsilon: ORT subgraph run introduces float32 rounding
  fuse_matmul_add             Epsilon: Gemm kernel may differ from MatMul+Add kernel
  fuse_matmul_add_3d          Same as above
  fuse_attention_qkv          Epsilon: Weight concat + split = same computation

Class 2 — Empirically Safe (no proof, need 50+ model study)
  cleanup_attention           Complex structural change — needs corpus verification
  simplify_shape_chains       Depends on shape inference correctness
  decompose_layernorm         Structurally equivalent but floating point order changes

Class 3 — Architecture-Dependent (safe on some models, unknown on others)
  (none identified yet — may emerge from M11 forensics)
```

**Target: Write this taxonomy as a 2-page section with proofs for Class 0.
This is the core of the ISSTA/ICSE submission.**

---

## Hard Gates (M11 Does Not Close Until All Pass)

| Gate | Threshold | Reason |
|------|-----------|--------|
| Forensics audit complete on 6 transformer models | All rows filled | Can't build passes without knowing what's there |
| fuse_matmul_add_3d fires on BERT | >5% node reduction | Closes the known M10 gap |
| At least one new pass fires on TinyLlama | Any non-zero reduction | Proves LLM optimization is possible |
| YOLOv8n tested | Transpose elimination proven on real YOLO | Closes the M10 deferred item |
| Accuracy verified on all new models | max_diff = 0.0 | Non-negotiable |
| Latency measured on 8+ new models | Table complete | Feeds Exp 21 correlation analysis |
| Attribution matrix updated | 17+ passes × 16+ models, no TBDs | Research foundation |
| Accuracy taxonomy draft written | Class 0 proofs complete, Class 1 bounded | First paper section |
| Node/latency correlation measured | Exp 21 complete | Publishable finding |
| Architecture fingerprint prototype | Can classify 8+ models correctly | Feeds CLI auto-selection |

---

## Updated Benchmark Table (Target State After M11)

```
| Model            | Nodes Before | Nodes After | Reduction | Latency Δ | max_diff |
|------------------|-------------|-------------|-----------|-----------|----------|
| MobileNetV2      | 105         | 105         | 0%        | -         | 0.00e+00 |
| EfficientNet-B0  | 288         | 239         | 17%       | +1.3%     | 0.00e+00 |
| ResNet-50        | 179         | 122         | 32%       | +1.3%     | 0.00e+00 |
| MobileNetV3-S    | 175         | 141         | 19%       | -0.1%     | 0.00e+00 |
| BERT-base        | 1453        | TBD (>5%)   | TBD       | TBD       | 0.00e+00 |
| DistilBERT       | 743         | TBD         | TBD       | TBD       | 0.00e+00 |
| RoBERTa-base     | 1453        | TBD         | TBD       | TBD       | 0.00e+00 |
| Whisper-tiny     | 453         | TBD         | TBD       | TBD       | 0.00e+00 |
| Whisper-base     | TBD         | TBD         | TBD       | TBD       | TBD      |
| GPT-2-Small      | TBD         | TBD         | TBD       | TBD       | TBD      |
| TinyLlama-1.1B   | TBD         | TBD         | TBD       | TBD       | TBD      |
| YOLOv8n          | TBD         | TBD         | TBD       | TBD       | TBD      |
| YOLOv8s          | TBD         | TBD         | TBD       | TBD       | TBD      |
| DeiT-Small       | TBD         | TBD         | TBD       | TBD       | TBD      |
| MobileViT-S      | TBD         | TBD         | TBD       | TBD       | TBD      |
| ViT-Base         | TBD         | TBD         | TBD       | TBD       | TBD      |
| Wav2Vec2-Base    | TBD         | TBD         | TBD       | TBD       | TBD      |
```

---

## Build Order

### Week 1 — Forensics First
1. Write `tools/graph_forensics.py`
2. Run Exp 11 on BERT-base, DistilBERT, RoBERTa, Whisper-tiny
3. Fill forensics table — this determines what to build
4. Download YOLOv8n, YOLOv8s — run Exp 15 (transpose deep dive, deferred from M10)
5. Decision gate: which new passes show non-zero in forensics?

### Week 2 — New Transformer Passes
1. Implement Pass 12 (fuse_matmul_add_3d) — the known fix
2. Run Exp 12: verify >5% reduction on BERT
3. Implement Pass 13/14 (noop elimination) IF forensics showed >10 instances
4. Implement Pass 17 (fold_cast_chains) IF forensics showed cast chains
5. Run updated pipeline on all 8 existing transformer models
6. Measure latency improvement

### Week 3 — New Models
1. Download and test GPT-2, DeiT, MobileViT, ViT-Base
2. Run Exp 17 (GPT-2 vs BERT divergence)
3. Run Exp 16 (hybrid vision transformer test)
4. Run Exp 22 (architecture fingerprinting prototype)
5. Download TinyLlama — run Exp 18
6. Download Wav2Vec2, Whisper-base

### Week 4 — Experiments + Research
1. Run Exp 19 (full attribution matrix, all 16 models)
2. Run Exp 20 (latency benchmark, all new models)
3. Run Exp 21 (node/latency correlation — the publishable one)
4. Implement Pass 15 (QKV fusion) IF Exp 14 feasibility study passes
5. Implement Pass 16 (decompose_layernorm) — TFLite prerequisite
6. Write accuracy taxonomy first draft (Class 0 proofs)
7. Update README benchmark table — all rows filled

---

## What Cracks in M11

If M11 goes as planned, three things become true that weren't true before:

**1. Transformer optimization is real, not theoretical.**
BERT goes from 0% → 15%+ node reduction. The attribution table has non-zero
entries in the transformer columns. The tool works on the model class that matters most.

**2. The LLM story starts.**
TinyLlama running through onnxforge with measurable improvement is the demo
that gets Qualcomm's attention. It's the model class ZETIC targets. It's the
model class on-device AI is racing toward.

**3. The first paper section exists.**
The accuracy taxonomy with Class 0 proofs is the core of a real academic contribution.
It answers a question nobody has formally answered: which ONNX optimizations are
provably safe by construction, and which require empirical verification?

---

## What Comes After M11

**M12 — TFLite Conversion Pipeline (Phase 2 begins)**
- decompose_layernorm (Pass 16) is the prerequisite — built in M11
- First end-to-end: ONNX → onnxforge → .tflite
- Test on: EfficientNet-B0, ResNet-50, BERT-base, Whisper-tiny
- Benchmark: conversion success rate before vs. after onnxforge preprocessing

**M13 — CoreML Conversion Pipeline**
- coremltools wrapping
- Mac M3 native inference benchmarks
- First cross-platform: same model, TFLite latency vs. CoreML latency

**Research Agenda**
- M11 Exp 21 (node/latency correlation) → short paper or workshop submission
- M11 accuracy taxonomy → ISSTA/ICSE submission draft
- M11 architecture fingerprinting → feeds into CLI auto-selection feature

M11 is where onnxforge stops being a vision model optimizer
and becomes a full-spectrum model optimization pipeline.
