# M9 — Validation, Real Models, and Attention Cleanup

**This milestone is a reckoning.**

M1–M8 built passes. M9 proves they actually work — on real models, with real numbers,
against adversarial inputs, with per-pass attribution showing exactly which pass did what.
No more "this will matter on BERT later." BERT is now. YOLOv8n is now. EfficientNet is now.

M9 does not complete until every experiment below has a filled-in result in the progress
tracker. Every blank is an unknown. Unknowns are not done.

---

## What You're Building

```
onnxslim/
├── passes/
│   └── cleanup_attention.py                 ← NEW — Tier 4 novel pass
├── tests/
│   ├── toy_models/
│   │   └── build_attention_model.py         ← NEW
│   ├── test_attention.py                    ← NEW
│   └── experiments/
│       ├── __init__.py
│       ├── exp_01_pass_attribution.py       ← which pass does what, per model
│       ├── exp_02_yolov8_transposes.py      ← prove M4 fires on real model
│       ├── exp_03_efficientnet_bn.py        ← prove M7 fires on real model
│       ├── exp_04_bert_full_pipeline.py     ← full BERT benchmark
│       ├── exp_05_conv_relu_ort.py          ← find ORT-safe Conv+ReLU approach
│       ├── exp_06_tolerance_sweep.py        ← find real accuracy bounds
│       ├── exp_07_pass_order_sensitivity.py ← does order matter?
│       └── exp_08_graph_inspector.py        ← dump op counts before/after
models/
├── bert-base-uncased.onnx                   ← REQUIRED
├── yolov8n.onnx                             ← REQUIRED
└── efficientnet-b0.onnx                     ← REQUIRED
```

---

## Part 1 — Fix Conv+ReLU (M8 Correction)

The M8 Conv+ReLU pass was implemented as a pattern detector only, with the reason
given as "ORT rejects unknown attributes." This must be investigated properly, not
assumed. ORT is our verification backbone — if ORT rejects something, verify.py breaks,
and we lose our accuracy guarantee. But "ORT might reject it" is not the same as
"ORT does reject it." Run the experiment.

### Experiment 05 — Find what ORT actually accepts

```python
# tests/experiments/exp_05_conv_relu_ort.py
"""
Experiment: Find the right approach for Conv+ReLU fusion that:
1. ORT can still run (verify.py continues to work)
2. Provides value to downstream converters (TFLite, CoreML)
3. Is provably correct numerically

Test four approaches. Document exactly which ones ORT accepts and what
the output difference is. This determines how we fix M8.
"""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper
import copy


def make_conv_relu_model():
    W = np.ones((2, 1, 3, 3), dtype=np.float32)
    X    = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])
    Y    = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 3, 3])
    conv = helper.make_node("Conv", ["X", "W"], ["conv_out"], kernel_shape=[3, 3])
    relu = helper.make_node("Relu", ["conv_out"], ["Y"])
    graph = helper.make_graph([conv, relu], "conv_relu", [X], [Y])
    graph.initializer.append(numpy_helper.from_array(W, "W"))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def approach_a(model):
    """Remove Relu, add custom activation attribute to Conv."""
    m = copy.deepcopy(model)
    relu = next(n for n in m.graph.node if n.op_type == "Relu")
    conv = next(n for n in m.graph.node if n.op_type == "Conv")
    conv.attribute.append(helper.make_attribute("activation", "Relu"))
    conv.output[0] = relu.output[0]
    m.graph.node.remove(relu)
    return m


def approach_b(model):
    """Keep Relu, add annotation-only attribute to Conv. Graph unchanged."""
    m = copy.deepcopy(model)
    for node in m.graph.node:
        if node.op_type == "Conv":
            conv_out = node.output[0]
            for n2 in m.graph.node:
                if n2.op_type == "Relu" and n2.input[0] == conv_out:
                    node.attribute.append(
                        helper.make_attribute("activation_hint", "Relu")
                    )
    return m


def approach_c(model):
    """Replace Relu with Clip(0, inf) — semantically identical, better TFLite compat."""
    m = copy.deepcopy(model)
    relu = next(n for n in m.graph.node if n.op_type == "Relu")
    min_t = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "clip_min")
    max_t = numpy_helper.from_array(np.array(3.4e38, dtype=np.float32), "clip_max")
    m.graph.initializer.extend([min_t, max_t])
    clip = helper.make_node(
        "Clip",
        inputs=[relu.input[0], "clip_min", "clip_max"],
        outputs=[relu.output[0]],
    )
    idx = list(m.graph.node).index(relu)
    m.graph.node.remove(relu)
    m.graph.node.insert(idx, clip)
    return m


def approach_d(model):
    """
    Two-model strategy: keep original for ORT verification,
    produce a separate export-only version with Relu removed.
    This approach separates ORT verification from converter output.
    """
    # For ORT: model unchanged
    ort_model = copy.deepcopy(model)
    # For converter: Relu removed with annotation
    export_model = approach_a(model)
    return ort_model, export_model


def test_ort(model, label):
    try:
        sess = ort.InferenceSession(model.SerializeToString())
        inp = {"X": np.random.randn(1, 1, 5, 5).astype(np.float32)}
        out = sess.run(None, inp)
        return True, out[0]
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    print("Experiment 05 — Conv+ReLU ORT Compatibility\n")

    baseline = make_conv_relu_model()
    ok, base_out = test_ort(baseline, "baseline")
    print(f"Baseline ORT: {'✓' if ok else '✗'}\n")

    experiments = [
        ("A — Relu removed, custom attr", approach_a),
        ("B — Relu kept, annotation only", approach_b),
        ("C — Relu → Clip(0, inf)", approach_c),
    ]

    for label, fn in experiments:
        m = fn(baseline)
        ok, out_or_err = test_ort(m, label)
        print(f"Approach {label}:")
        print(f"  ORT accepts: {'✓' if ok else '✗'}")
        if ok:
            diff = float(np.max(np.abs(base_out - out_or_err)))
            print(f"  max_diff vs baseline: {diff:.2e}")
        else:
            print(f"  ORT error: {str(out_or_err)[:300]}")
        print()

    print("Approach D — Two-model strategy:")
    ort_m, export_m = approach_d(baseline)
    ok_ort, out_ort = test_ort(ort_m, "D-ort")
    ok_exp, out_exp = test_ort(export_m, "D-export")
    print(f"  ORT model accepts: {'✓' if ok_ort else '✗'}")
    print(f"  Export model ORT:  {'✓' if ok_exp else '✗'}")
    if ok_ort and ok_exp:
        diff = float(np.max(np.abs(out_ort - out_exp)))
        print(f"  Diff between versions: {diff:.2e}")
```

**Record in progress tracker:**
```
Exp 05 — Conv+ReLU ORT Compatibility:
  Approach A (Relu removed, custom attr): ORT=___  max_diff=___
  Approach B (annotation, Relu kept):     ORT=___  max_diff=___
  Approach C (Clip replacement):          ORT=___  max_diff=___
  Approach D (two-model strategy):        ORT=___  diff=___
  Chosen approach: ___
  Reason: ___
  M8 updated: YES / NO
  New fuse_conv_relu behavior: ___
```

---

## Part 2 — Model Downloads (All Required)

Not optional. Run these before any experiments.
`do_constant_folding=False` everywhere — if PyTorch pre-folds, our benchmark is meaningless.

```bash
mkdir -p models tests/experiments

# YOLOv8n
pip install ultralytics
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', opset=13, dynamic=False, imgsz=640, simplify=False)
import shutil; shutil.move('yolov8n.onnx', 'models/yolov8n.onnx')
print('Done: models/yolov8n.onnx')
"

# EfficientNet-B0
pip install torchvision torch
python -c "
import torchvision, torch
model = torchvision.models.efficientnet_b0(weights='DEFAULT').eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'models/efficientnet-b0.onnx',
    opset_version=13, input_names=['input'], output_names=['output'],
    do_constant_folding=False)
print('Done: models/efficientnet-b0.onnx')
"

# BERT-base
pip install transformers torch
python -c "
from transformers import BertModel
import torch
model = BertModel.from_pretrained('bert-base-uncased').eval()
dummy = {'input_ids': torch.ones(1,128,dtype=torch.long),
         'attention_mask': torch.ones(1,128,dtype=torch.long)}
torch.onnx.export(model, (dummy,), 'models/bert-base-uncased.onnx',
    input_names=['input_ids','attention_mask'],
    output_names=['last_hidden_state'],
    dynamic_axes={'input_ids':{0:'batch',1:'seq'},
                  'attention_mask':{0:'batch',1:'seq'}},
    opset_version=14,
    do_constant_folding=False)
print('Done: models/bert-base-uncased.onnx')
"
```

---

## Part 3 — Graph Inspector

**Run this first on every model.** Before passing anything through our optimizer,
know exactly what ops are in it. This is how we avoid surprises like "MobileNetV2
has 0 BN nodes" discovered after building M7.

```python
# tests/experiments/exp_08_graph_inspector.py
"""
Inspect any ONNX model — full op inventory and pass relevance flags.
Usage: python tests/experiments/exp_08_graph_inspector.py models/yolov8n.onnx
       python tests/experiments/exp_08_graph_inspector.py  (runs all models)
"""
import sys
import os
import onnx
from collections import Counter


def inspect(model_path):
    model     = onnx.load(model_path)
    graph     = model.graph
    op_counts = Counter(n.op_type for n in graph.node)
    nodes     = len(graph.node)
    size_mb   = model.ByteSize() / 1024 / 1024
    n_inits   = len(graph.initializer)
    opset     = next((o.version for o in model.opset_import
                      if o.domain in ('', 'ai.onnx')), '?')

    print(f"\n{'='*58}")
    print(f"  {model_path}")
    print(f"{'='*58}")
    print(f"  Nodes: {nodes}   Initializers: {n_inits}   "
          f"Size: {size_mb:.1f}MB   Opset: {opset}")
    print(f"\n  Op inventory:")
    for op, count in op_counts.most_common():
        bar = '█' * min(count, 50)
        print(f"    {op:<28} {count:>5}  {bar}")

    print(f"\n  Pass relevance:")
    checks = [
        ("M3 unused_inits",       f"{n_inits} initializers"),
        ("M4 transposes",         f"{op_counts['Transpose']} Transpose"),
        ("M5 fold_constants",     f"{op_counts['Constant']} Constant"),
        ("M6 shape_chains",       f"{op_counts['Reshape']} Reshape  {op_counts['Shape']} Shape"),
        ("M7 conv_bn",            f"{op_counts['Conv']} Conv  {op_counts['BatchNormalization']} BN"),
        ("M8 conv_relu",          f"{op_counts['Relu']} Relu  {op_counts['Clip']} Clip"),
        ("M8 matmul_add→gemm",    f"{op_counts['MatMul']} MatMul  {op_counts['Add']} Add  {op_counts['Gemm']} Gemm"),
        ("M9 attention_cleanup",  f"{op_counts['Reshape']} Reshape  {op_counts['Transpose']} Transpose"),
    ]
    for label, detail in checks:
        print(f"    {label:<25} {detail}")

    print()
    return op_counts, nodes, size_mb


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else [
        "mobilenetv2-12.onnx",
        "models/yolov8n.onnx",
        "models/efficientnet-b0.onnx",
        "models/bert-base-uncased.onnx",
    ]
    for path in targets:
        if os.path.exists(path):
            inspect(path)
        else:
            print(f"\n⚠ Not found: {path}")
```

**Record in progress tracker — required for all 4 models:**
```
Exp 08 — Graph Inspector:

  MobileNetV2:  nodes=___  size=___MB
    Transpose=___  Constant=___  Reshape=___  Shape=___
    Conv=___  BN=___  Relu=___  Clip=___  MatMul=___  Add=___

  YOLOv8n:      nodes=___  size=___MB
    Transpose=___  Constant=___  Reshape=___  Shape=___
    Conv=___  BN=___  Relu=___  Clip=___  MatMul=___  Add=___

  EfficientNet: nodes=___  size=___MB
    Transpose=___  Constant=___  Reshape=___  Shape=___
    Conv=___  BN=___  Relu=___  Clip=___  MatMul=___  Add=___

  BERT-base:    nodes=___  size=___MB
    Transpose=___  Constant=___  Reshape=___  Shape=___
    Conv=___  BN=___  Relu=___  Clip=___  MatMul=___  Add=___
```

---

## Part 4 — Per-Pass Attribution

The single most important experiment. Runs every pass on every model independently
AND cumulatively, recording exactly how many nodes each pass removes. No more guessing.

```python
# tests/experiments/exp_01_pass_attribution.py
"""
Run each pass individually and cumulatively on each model.
Records: nodes removed per pass, accuracy preserved.
This is the honest accounting of what the optimizer actually does.

Usage: python tests/experiments/exp_01_pass_attribution.py
"""
import onnx
import os
from verify import verify


def all_passes():
    from passes.eliminate_dead_nodes import EliminateDeadNodes
    from passes.eliminate_identity_ops import EliminateIdentityOps
    from passes.eliminate_unused_initializers import EliminateUnusedInitializers
    from passes.eliminate_duplicate_constants import EliminateDuplicateConstants
    from passes.eliminate_redundant_transposes import EliminateRedundantTransposes
    from passes.fold_constants import FoldConstants
    from passes.simplify_shape_chains import SimplifyShapeChains
    from passes.fuse_conv_batchnorm import FuseConvBatchnorm
    from passes.fuse_conv_relu import FuseConvRelu
    from passes.fuse_matmul_add import FuseMatmulAdd
    from passes.cleanup_attention import CleanupAttention
    return [
        EliminateDeadNodes(), EliminateIdentityOps(),
        EliminateUnusedInitializers(), EliminateDuplicateConstants(),
        EliminateRedundantTransposes(), FoldConstants(),
        SimplifyShapeChains(), FuseConvBatchnorm(),
        FuseConvRelu(), FuseMatmulAdd(), CleanupAttention(),
    ]


def run_attribution(model_path, tolerance=1e-4):
    if not os.path.exists(model_path):
        print(f"  ⚠ Not found: {model_path}")
        return {}

    original = onnx.load(model_path)
    baseline_nodes = len(original.graph.node)
    baseline_size  = original.ByteSize() / 1024 / 1024

    print(f"\n{'='*65}")
    print(f"  {model_path}")
    print(f"  Baseline: {baseline_nodes} nodes  {baseline_size:.1f}MB")
    print(f"{'='*65}")
    print(f"  {'Pass':<38} {'isolated':>9} {'cumul':>7}  {'accuracy'}")
    print(f"  {'-'*38}  {'-'*9}  {'-'*7}  {'-'*12}")

    cumulative  = onnx.load(model_path)
    attribution = {}

    for p in all_passes():
        nodes_before_cumul = len(cumulative.graph.node)

        # Isolated run — just this one pass on the original
        try:
            m_isolated = p.run(onnx.load(model_path))
            isolated_delta = baseline_nodes - len(m_isolated.graph.node)
        except Exception as e:
            isolated_delta = f"ERR"

        # Cumulative run — this pass on top of all previous
        try:
            cumulative = p.run(cumulative)
            nodes_after_cumul = len(cumulative.graph.node)
            cumul_delta = nodes_before_cumul - nodes_after_cumul
        except Exception as e:
            cumul_delta = f"ERR:{str(e)[:20]}"
            nodes_after_cumul = nodes_before_cumul

        # Accuracy check on cumulative model
        try:
            rpt = verify(original, cumulative, n_samples=3, tolerance=tolerance)
            acc = f"✓ {rpt.max_diff:.1e}"
        except Exception as e:
            acc = f"✗ FAIL"

        iso_str   = f"{-isolated_delta:+d}" if isinstance(isolated_delta, int) else str(isolated_delta)
        cumul_str = f"{-cumul_delta:+d}"    if isinstance(cumul_delta, int) else str(cumul_delta)
        print(f"  {p.name:<38} {iso_str:>9}  {cumul_str:>7}  {acc}")
        attribution[p.name] = {"isolated": isolated_delta, "cumulative": cumul_delta}

    final = len(cumulative.graph.node)
    final_size = cumulative.ByteSize() / 1024 / 1024
    print(f"\n  TOTAL: {baseline_nodes} → {final} ({baseline_nodes - final:+d} nodes)")
    print(f"  SIZE:  {baseline_size:.1f}MB → {final_size:.1f}MB")
    return attribution


if __name__ == "__main__":
    models = [
        ("mobilenetv2-12.onnx",           1e-5),
        ("models/yolov8n.onnx",            1e-4),
        ("models/efficientnet-b0.onnx",    1e-4),
        ("models/bert-base-uncased.onnx",  1e-4),
    ]
    all_results = {}
    for path, tol in models:
        all_results[path] = run_attribution(path, tol)

    # Summary table
    print(f"\n\n{'='*65}")
    print(f"  SUMMARY — Isolated node reduction per pass per model")
    print(f"{'='*65}")
    model_names = ["MobileNetV2", "YOLOv8n", "EfficientNet", "BERT"]
    print(f"  {'Pass':<38}", end="")
    for name in model_names:
        print(f" {name:>12}", end="")
    print()
    print(f"  {'-'*38}", end="")
    for _ in model_names:
        print(f" {'':>12}", end="")
    print()

    pass_names = [p.name for p in all_passes()]
    for pname in pass_names:
        print(f"  {pname:<38}", end="")
        for path, _ in models:
            res = all_results.get(path, {}).get(pname, {})
            iso = res.get("isolated", "?")
            val = f"{-iso:+d}" if isinstance(iso, int) else str(iso)
            print(f" {val:>12}", end="")
        print()
```

**Record in progress tracker — fill every cell:**
```
Exp 01 — Pass Attribution (isolated Δ nodes):

Pass                            | MobileNetV2 | YOLOv8n | EfficientNet | BERT
--------------------------------|-------------|---------|--------------|------
eliminate_dead_nodes            |             |         |              |
eliminate_identity_ops          |             |         |              |
eliminate_unused_initializers   |             |         |              |
eliminate_duplicate_constants   |     -68     |         |              |
eliminate_redundant_transposes  |       0     |         |              |
fold_constants                  |       0     |         |              |
simplify_shape_chains           |       0     |         |              |
fuse_conv_batchnorm             |       0     |         |              |
fuse_conv_relu                  |       0     |         |              |
fuse_matmul_add                 |       0     |         |              |
cleanup_attention               |       0     |         |              |
TOTAL                           |             |         |              |

Any pass showing 0 across ALL 4 models → investigate before M10.
```

---

## Part 5 — YOLOv8n Transpose Experiment

Proves M4 fires on a real model. YOLOv8n is NCHW exported to ONNX — it must have
transpose patterns. If it doesn't, something is wrong with either the export or the pass.

```python
# tests/experiments/exp_02_yolov8_transposes.py
"""
Experiment: Prove eliminate_redundant_transposes fires on YOLOv8n.
Includes detailed pattern analysis: which perms cancel, which merge.
"""
import onnx
from verify import verify
from passes.eliminate_redundant_transposes import EliminateRedundantTransposes


def analyze_transpose_patterns(model):
    graph = model.graph
    output_to_consumers = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                output_to_consumers.setdefault(inp, []).append(node)

    transpose_nodes = [n for n in graph.node if n.op_type == "Transpose"]
    pairs = {"cancelling": [], "mergeable": [], "isolated": []}
    seen  = set()

    for node in transpose_nodes:
        if id(node) in seen:
            continue
        out = node.output[0]
        consumers = output_to_consumers.get(out, [])
        if len(consumers) == 1 and consumers[0].op_type == "Transpose":
            nxt = consumers[0]
            p1 = list(next(a.ints for a in node.attribute if a.name == "perm"))
            p2 = list(next(a.ints for a in nxt.attribute  if a.name == "perm"))
            composed = [p1[p2[i]] for i in range(len(p2))]
            if composed == list(range(len(composed))):
                pairs["cancelling"].append((p1, p2))
            else:
                pairs["mergeable"].append((p1, p2, composed))
            seen.add(id(node)); seen.add(id(nxt))
        else:
            pairs["isolated"].append(node)

    return transpose_nodes, pairs


def run():
    path = "models/yolov8n.onnx"
    print("Experiment 02 — YOLOv8n Transpose Analysis\n")

    original = onnx.load(path)
    transpose_nodes, pairs = analyze_transpose_patterns(original)

    print(f"BEFORE:")
    print(f"  Total Transpose nodes:   {len(transpose_nodes)}")
    print(f"  Cancelling pairs:        {len(pairs['cancelling'])}")
    print(f"  Mergeable pairs:         {len(pairs['mergeable'])}")
    print(f"  Isolated (no pair):      {len(pairs['isolated'])}")
    if pairs["cancelling"]:
        print(f"  Example cancel: {pairs['cancelling'][0]}")
    if pairs["mergeable"]:
        print(f"  Example merge:  perm1={pairs['mergeable'][0][0]}  "
              f"perm2={pairs['mergeable'][0][1]}  composed={pairs['mergeable'][0][2]}")

    model     = onnx.load(path)
    optimized = EliminateRedundantTransposes().run(model)
    _, pairs_after = analyze_transpose_patterns(optimized)
    t_after  = sum(1 for n in optimized.graph.node if n.op_type == "Transpose")
    removed  = len(original.graph.node) - len(optimized.graph.node)

    print(f"\nAFTER:")
    print(f"  Transpose nodes: {t_after}")
    print(f"  Cancelling pairs remaining: {len(pairs_after['cancelling'])}")
    print(f"  Nodes removed: {removed}")

    report = verify(original, optimized, n_samples=5, tolerance=1e-4)
    print(f"\nAccuracy: max_diff={report.max_diff:.2e}  "
          f"{'✓' if report.passed else '✗ FAILED'}")

    if removed == 0:
        print("\n⚠ ZERO nodes removed — investigate:")
        print("  1. Check if YOLOv8n was exported with simplify=True (pre-simplified)")
        print("  2. Check if all Transpose pairs have multiple consumers (branching)")
        print("  3. Manually inspect: python -c \"import onnx; m=onnx.load('models/yolov8n.onnx'); "
              "[print(n.op_type, [list(a.ints) for a in n.attribute if a.name=='perm']) "
              "for n in m.graph.node if n.op_type=='Transpose'][:10]\"")


if __name__ == "__main__":
    run()
```

**Record in progress tracker:**
```
Exp 02 — YOLOv8n Transpose:
  Total Transpose before:  ___
  Total Transpose after:   ___
  Cancelling pairs found:  ___
  Mergeable pairs found:   ___
  Nodes removed:           ___
  max_diff:                ___
  Pass fired?              YES / NO
  If NO — root cause:      ___
  Action taken:            ___
```

---

## Part 6 — EfficientNet BN Fusion Experiment

```python
# tests/experiments/exp_03_efficientnet_bn.py
"""
Experiment: Prove fuse_conv_batchnorm fires on EfficientNet-B0.
EfficientNet is dense with Conv+BN — all should fuse.
Also checks for unfused BN nodes and explains why.
"""
import onnx
from verify import verify
from passes.fuse_conv_batchnorm import FuseConvBatchnorm


def count_ops(model, *ops):
    return {op: sum(1 for n in model.graph.node if n.op_type == op) for op in ops}


def find_unfused_bn(model):
    """Find BN nodes that don't immediately follow a Conv."""
    graph   = model.graph
    outputs = {out: node for node in graph.node for out in node.output}
    unfused = []
    for node in graph.node:
        if node.op_type == "BatchNormalization":
            producer = outputs.get(node.input[0])
            if producer is None or producer.op_type != "Conv":
                unfused.append((node.name, producer.op_type if producer else "graph_input"))
    return unfused


def run():
    path = "models/efficientnet-b0.onnx"
    print("Experiment 03 — EfficientNet-B0 Conv+BN Fusion\n")

    original = onnx.load(path)
    b4 = count_ops(original, "Conv", "BatchNormalization", "Relu", "Clip", "Sigmoid")
    size_b4 = original.ByteSize() / 1024 / 1024

    print("BEFORE:")
    for op, count in b4.items():
        print(f"  {op:<25} {count}")
    print(f"  Total nodes: {len(original.graph.node)}")
    print(f"  Size: {size_b4:.2f}MB")

    # Check for unfused BN before running (expect 0)
    unfused_before = find_unfused_bn(original)
    if unfused_before:
        print(f"\n  BN nodes not after Conv (pre-existing): {len(unfused_before)}")
        for name, prev in unfused_before[:3]:
            print(f"    BN '{name}' — producer is: {prev}")

    model     = onnx.load(path)
    optimized = FuseConvBatchnorm().run(model)
    af = count_ops(optimized, "Conv", "BatchNormalization", "Relu", "Clip", "Sigmoid")
    size_af = optimized.ByteSize() / 1024 / 1024

    print("\nAFTER:")
    for op in b4:
        delta = b4[op] - af[op]
        marker = f"  (-{delta})" if delta > 0 else ""
        print(f"  {op:<25} {af[op]}{marker}")
    print(f"  Total nodes: {len(optimized.graph.node)}")
    print(f"  Size: {size_af:.2f}MB  ({size_b4 - size_af:+.2f}MB)")

    # Check for remaining unfused BN
    unfused_after = find_unfused_bn(optimized)
    if unfused_after:
        print(f"\n  ⚠ {len(unfused_after)} BN nodes NOT fused:")
        for name, prev in unfused_after[:5]:
            print(f"    BN '{name}' — producer is: {prev}")

    report = verify(original, optimized, n_samples=5, tolerance=1e-4)
    print(f"\nAccuracy: max_diff={report.max_diff:.2e}  "
          f"{'✓' if report.passed else '✗ FAILED'}")

    if af["BatchNormalization"] == 0:
        print("\n✓ All BN nodes fused.")
    else:
        print(f"\n⚠ {af['BatchNormalization']} BN nodes remain — investigate above.")


if __name__ == "__main__":
    run()
```

**Record in progress tracker:**
```
Exp 03 — EfficientNet-B0 BN Fusion:
  BN before/after:      ___ → ___
  Conv before/after:    ___ → ___
  Nodes removed:        ___
  Size before/after:    ___MB → ___MB
  max_diff:             ___
  All BN fused?         YES / NO
  Unfused BN count:     ___
  Unfused BN reason:    ___
```

---

## Part 7 — Tolerance Sweep

Stop assuming tolerances. Measure them.

```python
# tests/experiments/exp_06_tolerance_sweep.py
"""
Experiment: Measure actual max_diff for each model across 20 random seeds.
Gives the real accuracy bounds — not assumed ones.
"""
import numpy as np
import onnx
import onnxruntime as ort
import os


def gen_inputs(model, seed):
    np.random.seed(seed)
    feed = {}
    for inp in model.graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value > 0 else 1)
        dtype = inp.type.tensor_type.elem_type
        if dtype == 1:
            feed[inp.name] = np.random.randn(*shape).astype(np.float32)
        elif dtype == 7:
            feed[inp.name] = np.random.randint(0, 100, shape).astype(np.int64)
        else:
            feed[inp.name] = np.random.randn(*shape).astype(np.float32)
    return feed


def sweep(orig_path, opt_path, n=20):
    orig = onnx.load(orig_path)
    opt  = onnx.load(opt_path)
    sess_orig = ort.InferenceSession(orig.SerializeToString())
    sess_opt  = ort.InferenceSession(opt.SerializeToString())
    out_names = [o.name for o in orig.graph.output]
    diffs = []
    for seed in range(n):
        feed = gen_inputs(orig, seed)
        a = sess_orig.run(out_names, feed)
        b = sess_opt.run(out_names, feed)
        for x, y in zip(a, b):
            diffs.append(float(np.max(np.abs(np.array(x) - np.array(y)))))
    return {
        "min": min(diffs), "max": max(diffs),
        "mean": float(np.mean(diffs)), "p99": float(np.percentile(diffs, 99)),
        "all_zero": all(d == 0.0 for d in diffs),
    }


if __name__ == "__main__":
    from optimizer import optimize
    from passes.eliminate_dead_nodes import EliminateDeadNodes
    from passes.eliminate_identity_ops import EliminateIdentityOps
    from passes.eliminate_unused_initializers import EliminateUnusedInitializers
    from passes.eliminate_duplicate_constants import EliminateDuplicateConstants
    from passes.eliminate_redundant_transposes import EliminateRedundantTransposes
    from passes.fold_constants import FoldConstants
    from passes.simplify_shape_chains import SimplifyShapeChains
    from passes.fuse_conv_batchnorm import FuseConvBatchnorm
    from passes.fuse_conv_relu import FuseConvRelu
    from passes.fuse_matmul_add import FuseMatmulAdd
    from passes.cleanup_attention import CleanupAttention

    passes = [
        EliminateDeadNodes(), EliminateIdentityOps(),
        EliminateUnusedInitializers(), EliminateDuplicateConstants(),
        EliminateRedundantTransposes(), FoldConstants(),
        SimplifyShapeChains(), FuseConvBatchnorm(),
        FuseConvRelu(), FuseMatmulAdd(), CleanupAttention(),
    ]

    pairs = [
        ("mobilenetv2-12.onnx",          "models/mobilenetv2-opt.onnx"),
        ("models/yolov8n.onnx",           "models/yolov8n-opt.onnx"),
        ("models/efficientnet-b0.onnx",   "models/efficientnet-opt.onnx"),
        ("models/bert-base-uncased.onnx", "models/bert-opt.onnx"),
    ]

    print(f"\nExperiment 06 — Tolerance Sweep (n=20 seeds)\n")
    print(f"{'Model':<22} {'min':>10} {'max':>10} {'mean':>10} {'p99':>10} {'all_zero':>10}")
    print("-" * 76)

    for orig, opt in pairs:
        if not os.path.exists(orig):
            print(f"  ⚠ Missing: {orig}")
            continue
        if not os.path.exists(opt):
            optimize(orig, opt, passes=passes, verify_each_pass=False, n_verify_samples=1)
        r = sweep(orig, opt, n=20)
        name = os.path.basename(orig).replace(".onnx", "")
        print(f"{name:<22} {r['min']:>10.2e} {r['max']:>10.2e} "
              f"{r['mean']:>10.2e} {r['p99']:>10.2e} {str(r['all_zero']):>10}")
```

**Record in progress tracker:**
```
Exp 06 — Tolerance Sweep (20 seeds):
  Model          | min      | max      | mean     | p99      | all_zero
  MobileNetV2    |          |          |          |          |
  YOLOv8n        |          |          |          |          |
  EfficientNet   |          |          |          |          |
  BERT-base      |          |          |          |          |

  Recommended tolerance per model (use p99 * 10x safety margin):
    MobileNetV2:      ___
    YOLOv8n:          ___
    EfficientNet:     ___
    BERT-base:        ___
```

---

## Part 8 — Pass Order Sensitivity

```python
# tests/experiments/exp_07_pass_order_sensitivity.py
"""
Experiment: Does pass order affect the final node count?
Shuffles pass order 5 times, compares to canonical order.
Expected: canonical is at least as good as shuffled; some orders are worse.
"""
import random
import onnx
import os


def get_passes():
    from passes.eliminate_dead_nodes import EliminateDeadNodes
    from passes.eliminate_identity_ops import EliminateIdentityOps
    from passes.eliminate_unused_initializers import EliminateUnusedInitializers
    from passes.eliminate_duplicate_constants import EliminateDuplicateConstants
    from passes.eliminate_redundant_transposes import EliminateRedundantTransposes
    from passes.fold_constants import FoldConstants
    from passes.simplify_shape_chains import SimplifyShapeChains
    from passes.fuse_conv_batchnorm import FuseConvBatchnorm
    from passes.fuse_conv_relu import FuseConvRelu
    from passes.fuse_matmul_add import FuseMatmulAdd
    from passes.cleanup_attention import CleanupAttention
    return [
        EliminateDeadNodes(), EliminateIdentityOps(),
        EliminateUnusedInitializers(), EliminateDuplicateConstants(),
        EliminateRedundantTransposes(), FoldConstants(),
        SimplifyShapeChains(), FuseConvBatchnorm(),
        FuseConvRelu(), FuseMatmulAdd(), CleanupAttention(),
    ]


def run_order(model_path, passes):
    model = onnx.load(model_path)
    for p in passes:
        try:
            model = p.run(model)
        except Exception:
            pass
    return len(model.graph.node)


if __name__ == "__main__":
    models = [
        "mobilenetv2-12.onnx",
        "models/bert-base-uncased.onnx",
    ]
    for path in models:
        if not os.path.exists(path):
            continue
        print(f"\nModel: {path}")
        canonical = run_order(path, get_passes())
        print(f"  Canonical order:  {canonical} nodes")
        shuffled_results = []
        for i in range(5):
            p = get_passes()
            random.shuffle(p)
            count = run_order(path, p)
            shuffled_results.append(count)
            names = "→".join(x.name[:8] for x in p[:4]) + "→..."
            print(f"  Shuffle {i+1}:        {count} nodes  ({names})")
        best_shuffle = min(shuffled_results)
        print(f"  Canonical best?   {'YES' if canonical <= best_shuffle else 'NO — shuffle found ' + str(best_shuffle)}")
        print(f"  Order sensitive?  {'YES' if len(set(shuffled_results)) > 1 else 'NO'}")
```

**Record in progress tracker:**
```
Exp 07 — Pass Order Sensitivity:
  MobileNetV2:
    Canonical: ___ nodes
    Shuffled:  ___, ___, ___, ___, ___
    Sensitive: YES / NO
    Canonical optimal: YES / NO

  BERT-base:
    Canonical: ___ nodes
    Shuffled:  ___, ___, ___, ___, ___
    Sensitive: YES / NO
    Canonical optimal: YES / NO

  Finding: ___
  Action (if canonical not optimal): ___
```

---

## Part 9 — Full BERT Pipeline Experiment

```python
# tests/experiments/exp_04_bert_full_pipeline.py
"""
Full BERT benchmark — the number that goes in the README.
Runs the complete 11-pass pipeline with detailed per-op before/after reporting.
"""
import onnx
import os
import time
from collections import Counter
from verify import verify
from optimizer import optimize


def run():
    bert_path = "models/bert-base-uncased.onnx"
    bert_opt  = "models/bert-base-uncased-opt.onnx"

    if not os.path.exists(bert_path):
        print("⚠ BERT not found. Export it first.")
        return

    from passes.eliminate_dead_nodes import EliminateDeadNodes
    from passes.eliminate_identity_ops import EliminateIdentityOps
    from passes.eliminate_unused_initializers import EliminateUnusedInitializers
    from passes.eliminate_duplicate_constants import EliminateDuplicateConstants
    from passes.eliminate_redundant_transposes import EliminateRedundantTransposes
    from passes.fold_constants import FoldConstants
    from passes.simplify_shape_chains import SimplifyShapeChains
    from passes.fuse_conv_batchnorm import FuseConvBatchnorm
    from passes.fuse_conv_relu import FuseConvRelu
    from passes.fuse_matmul_add import FuseMatmulAdd
    from passes.cleanup_attention import CleanupAttention

    passes = [
        EliminateDeadNodes(), EliminateIdentityOps(),
        EliminateUnusedInitializers(), EliminateDuplicateConstants(),
        EliminateRedundantTransposes(), FoldConstants(),
        SimplifyShapeChains(), FuseConvBatchnorm(),
        FuseConvRelu(), FuseMatmulAdd(), CleanupAttention(),
    ]

    original  = onnx.load(bert_path)
    ops_before = Counter(n.op_type for n in original.graph.node)
    nodes_before = len(original.graph.node)
    size_before  = original.ByteSize() / 1024 / 1024

    print("Experiment 04 — Full BERT Pipeline\n")
    print(f"Nodes before:  {nodes_before}")
    print(f"Size before:   {size_before:.2f}MB")

    t0 = time.time()
    report = optimize(bert_path, bert_opt, passes=passes,
                      verify_each_pass=True, n_verify_samples=3)
    elapsed = time.time() - t0

    optimized = onnx.load(bert_opt)
    ops_after = Counter(n.op_type for n in optimized.graph.node)
    nodes_after = len(optimized.graph.node)
    size_after  = optimized.ByteSize() / 1024 / 1024

    print(f"\nNodes after:   {nodes_after}  (-{nodes_before - nodes_after})")
    print(f"Size after:    {size_after:.2f}MB  ({size_before - size_after:+.2f}MB)")
    print(f"Time:          {elapsed:.1f}s")

    print(f"\nKey op deltas:")
    track = ["Reshape","Transpose","MatMul","Gemm","Add","Constant",
             "Shape","Gather","Unsqueeze","Concat","BatchNormalization"]
    for op in track:
        b = ops_before.get(op, 0)
        a = ops_after.get(op, 0)
        if b > 0 or a > 0:
            print(f"  {op:<25} {b:>5} → {a:>5}  ({b-a:+d})")

    acc = verify(original, optimized, n_samples=10, tolerance=1e-4)
    print(f"\nAccuracy: max_diff={acc.max_diff:.2e}  "
          f"{'✓ PASSED' if acc.passed else '✗ FAILED'}")

    pct = (nodes_before - nodes_after) / nodes_before * 100
    print(f"\n=== README BENCHMARK TABLE ===")
    print(f"| Metric         | Before       | After        | Delta         |")
    print(f"|----------------|--------------|--------------|---------------|")
    print(f"| Nodes          | {nodes_before:<12} | {nodes_after:<12} | {nodes_before-nodes_after:<13} |")
    print(f"| Node reduction | -            | -            | {pct:.1f}%          |")
    print(f"| Size (MB)      | {size_before:<12.2f} | {size_after:<12.2f} | {size_before-size_after:<13.2f} |")
    print(f"| max_diff       | -            | {acc.max_diff:<12.2e} | -             |")
    print(f"| Pipeline time  | -            | {elapsed:<12.1f} | -             |")


if __name__ == "__main__":
    run()
```

**Record in progress tracker — this is your README table:**
```
Exp 04 — BERT Full Pipeline:
  Nodes before:    ___
  Nodes after:     ___
  Node reduction:  ___ (___ %)
  Size before:     ___ MB
  Size after:      ___ MB
  max_diff:        ___
  Pipeline time:   ___ s

  Key op deltas:
    Reshape:    ___ → ___  (-___)
    Transpose:  ___ → ___  (-___)
    MatMul:     ___ → ___  (-___)
    Gemm:       ___ → ___  (+___)
    Constant:   ___ → ___  (-___)
    Shape:      ___ → ___  (-___)
```

---

## Part 10 — cleanup_attention Pass Implementation

Now implement the pass. The experiments above tell you what patterns actually exist.
Don't guess — look at your Exp 08 results for BERT before writing the pattern matchers.

See the pass code in the previous M9 version — it remains unchanged. Copy it in as-is.
After the experiments run, you may find additional patterns to add.

Build toy models and run tests as specified in the original M9 plan.

---

## Run Order

```bash
# 0. Download all models
# (run download scripts from Part 2)

# 1. Inspect all models first — know what you're dealing with
python tests/experiments/exp_08_graph_inspector.py

# 2. Fix Conv+ReLU — determine the right approach
python tests/experiments/exp_05_conv_relu_ort.py

# 3. Build attention toy models and implement cleanup_attention
python tests/toy_models/build_attention_model.py
# (implement passes/cleanup_attention.py)
python tests/test_attention.py

# 4. Run per-pass attribution on all models
python tests/experiments/exp_01_pass_attribution.py

# 5. Targeted model experiments
python tests/experiments/exp_02_yolov8_transposes.py
python tests/experiments/exp_03_efficientnet_bn.py

# 6. Full BERT benchmark
python tests/experiments/exp_04_bert_full_pipeline.py

# 7. Tolerance sweep
python tests/experiments/exp_06_tolerance_sweep.py

# 8. Pass order sensitivity
python tests/experiments/exp_07_pass_order_sensitivity.py
```

---

## Definition of Done — Hard Gates

M9 is NOT complete until ALL of these are true:

**Toy model tests:**
- [ ] `test_consecutive_reshape` → 2 Reshape → 1 ✓
- [ ] `test_identity_reshape_attention` → Reshape removed ✓
- [ ] `test_branching_reshape` → 4 nodes stay 4 ✓

**Models:**
- [ ] YOLOv8n downloaded and inspected
- [ ] EfficientNet-B0 downloaded and inspected
- [ ] BERT-base exported with `do_constant_folding=False` and inspected

**Experiments — every field filled in progress tracker:**
- [ ] Exp 01 — attribution table complete for all 4 models
- [ ] Exp 02 — YOLOv8n transpose result recorded (did M4 fire? if not, why?)
- [ ] Exp 03 — EfficientNet BN result recorded (did M7 fire? if not, why?)
- [ ] Exp 04 — BERT benchmark table recorded
- [ ] Exp 05 — Conv+ReLU approach chosen, M8 updated accordingly
- [ ] Exp 06 — tolerance sweep complete, per-model tolerances documented
- [ ] Exp 07 — pass order sensitivity measured

**Hard gates — non-negotiable:**
- [ ] At least one model shows >10% node reduction from onnxslim passes
- [ ] BERT accuracy verified within documented tolerance
- [ ] Every pass that shows 0 reduction across ALL 4 models has a documented reason
- [ ] Conv+ReLU pass does something real (not pattern detection only)

---

## Progress Tracker Template

Copy this into your progress tracker and fill every field:

```
M9 Progress
============================================================
Date started: ___
Date completed: ___

MODELS DOWNLOADED:
  YOLOv8n:         YES/NO   size=___MB   export flags: ___
  EfficientNet:    YES/NO   size=___MB   export flags: ___
  BERT-base:       YES/NO   size=___MB   do_constant_folding=False? YES/NO

EXP 08 — GRAPH INSPECTOR:
  [paste full inspector output for each model]

EXP 05 — CONV+RELU FIX:
  Approach chosen: ___
  ORT accepts: ___   max_diff: ___
  M8 updated: YES/NO
  New behavior: ___

EXP 01 — PASS ATTRIBUTION:
  [paste full attribution table]

EXP 02 — YOLOV8N TRANSPOSES:
  Transpose before/after: ___ / ___
  Nodes removed: ___   Pass fired: YES/NO
  If NO: ___

EXP 03 — EFFICIENTNET BN:
  BN before/after: ___ / ___
  Size before/after: ___ / ___   Pass fired: YES/NO
  If NO: ___

EXP 04 — BERT FULL PIPELINE:
  Nodes: ___ → ___ (-___ / -___%)
  Size: ___MB → ___MB
  max_diff: ___   Accuracy: PASS/FAIL
  Time: ___s
  Key deltas: Reshape:___→___  Transpose:___→___  MatMul:___→___  Gemm:___→___

EXP 06 — TOLERANCE SWEEP:
  MobileNetV2:   min=___  max=___  p99=___  recommended=___
  YOLOv8n:       min=___  max=___  p99=___  recommended=___
  EfficientNet:  min=___  max=___  p99=___  recommended=___
  BERT:          min=___  max=___  p99=___  recommended=___

EXP 07 — PASS ORDER:
  BERT canonical: ___ nodes
  BERT shuffled:  ___, ___, ___, ___, ___
  Order sensitive: YES/NO   Canonical optimal: YES/NO
  Action: ___

ATTENTION PASS:
  test_consecutive_reshape: ___
  test_identity_reshape:    ___
  test_branching_reshape:   ___
  BERT Reshape before/after: ___ / ___
  BERT Transpose before/after: ___ / ___

HARD GATES:
  >10% node reduction on any model: YES (___) / NO
  BERT accuracy verified: YES (___) / NO
  Zero-impact passes with no explanation: ___
  Conv+ReLU now does real work: YES / NO
============================================================
```
