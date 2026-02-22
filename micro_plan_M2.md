# M2 — eliminate_dead_nodes + eliminate_identity_ops

**Goal:** Implement the first two real passes. Node count drops. Verify still passes. You now have a working optimization pipeline, not just a scaffold.

---

## What You're Building

```
onnxslim/
├── passes/
│   ├── __init__.py              ← update to register new passes
│   ├── base_pass.py             ← unchanged
│   ├── eliminate_dead_nodes.py  ← NEW
│   └── eliminate_identity_ops.py← NEW
├── optimizer.py                 ← update registered_passes list
└── tests/
    └── test_mobilenetv2.py      ← NEW — first real test
```

---

## Background — What These Passes Do

### eliminate_dead_nodes

A "dead" node is one whose outputs are never consumed — not by any other node, and not as a graph output. They exist as export artifacts, especially in HuggingFace models.

```
Graph outputs: [output_0]

Node A → output_0   ← LIVE (feeds graph output)
Node B → output_1   ← DEAD (output_1 used by nothing)
Node C → output_0   ← LIVE
         ↑ also uses output of Node B as input — but Node B is still dead
         because deadness is about outputs, not inputs
```

**Algorithm:** reverse BFS/DFS from graph outputs. Mark every node reachable. Remove unmarked ones.

### eliminate_identity_ops

`Identity` nodes simply copy their input to their output. They're inserted by exporters as shape/type annotation artifacts and do nothing at runtime.

```
Before:
  Conv → Identity → ReLU

After:
  Conv → ReLU
```

**Algorithm:** For every Identity node, find all nodes that use its output and rewire them to use its input directly. Remove the Identity node.

---

## Step-by-Step Tasks

### 1. `passes/eliminate_dead_nodes.py`

```python
import onnx
from .base_pass import BasePass

class EliminateDeadNodes(BasePass):

    @property
    def name(self) -> str:
        return "eliminate_dead_nodes"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        graph = model.graph

        # Step 1: collect all graph output names — these are always "live"
        live_outputs = set(o.name for o in graph.output)

        # Step 2: build a map from output_name → node (so we can trace backwards)
        output_to_node = {}
        for node in graph.node:
            for out in node.output:
                if out:  # output names can be empty strings in some exports
                    output_to_node[out] = node

        # Step 3: BFS backwards from graph outputs to find all live nodes
        live_nodes = set()
        queue = list(live_outputs)

        while queue:
            name = queue.pop()
            if name not in output_to_node:
                continue  # it's an initializer or graph input, not a node output
            node = output_to_node[name]
            node_id = id(node)
            if node_id in live_nodes:
                continue
            live_nodes.add(node_id)
            # this node's inputs may come from other nodes — trace them too
            for inp in node.input:
                if inp:
                    queue.append(inp)

        # Step 4: remove nodes not in live set
        dead = [n for n in graph.node if id(n) not in live_nodes]
        for node in dead:
            graph.node.remove(node)

        return model
```

---

### 2. `passes/eliminate_identity_ops.py`

```python
import onnx
from .base_pass import BasePass

class EliminateIdentityOps(BasePass):

    @property
    def name(self) -> str:
        return "eliminate_identity_ops"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        graph = model.graph

        # Build map: identity_output_name → identity_input_name
        # Only for Identity nodes
        remap = {}
        identity_nodes = []

        for node in graph.node:
            if node.op_type == "Identity":
                if node.input[0] and node.output[0]:
                    remap[node.output[0]] = node.input[0]
                    identity_nodes.append(node)

        if not remap:
            return model  # nothing to do

        # Resolve chains: Identity → Identity → Identity
        # e.g. A → Identity(out=B) → Identity(out=C)
        # remap[C] = B, remap[B] = A → we want remap[C] = A
        def resolve(name):
            visited = set()
            while name in remap and name not in visited:
                visited.add(name)
                name = remap[name]
            return name

        # Rewrite all node inputs that reference an identity output
        for node in graph.node:
            if node in identity_nodes:
                continue
            for i, inp in enumerate(node.input):
                if inp in remap:
                    node.input[i] = resolve(inp)

        # Rewrite graph outputs too
        for output in graph.output:
            if output.name in remap:
                output.name = resolve(output.name)

        # Remove identity nodes
        for node in identity_nodes:
            graph.node.remove(node)

        return model
```

---

### 3. Update `passes/__init__.py`

```python
from .eliminate_dead_nodes import EliminateDeadNodes
from .eliminate_identity_ops import EliminateIdentityOps

__all__ = ["EliminateDeadNodes", "EliminateIdentityOps"]
```

---

### 4. Update `optimizer.py` — register the passes

Replace the `registered_passes = []` block in `__main__` with:

```python
from passes import EliminateDeadNodes, EliminateIdentityOps

# M2: first two real passes
registered_passes = [
    EliminateDeadNodes(),
    EliminateIdentityOps(),
]
```

---

### 5. `tests/test_mobilenetv2.py`

Create the `tests/` directory and add the first real test file.

```python
"""
M2 Test — MobileNetV2
Verifies that eliminate_dead_nodes and eliminate_identity_ops:
  1. Reduce or maintain node count (never increase)
  2. Preserve accuracy within tolerance
  3. Produce a valid ONNX graph
"""
import onnx
import onnxruntime as ort
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optimizer import optimize
from verify import verify
from passes import EliminateDeadNodes, EliminateIdentityOps

MODEL_PATH = "mobilenetv2-12.onnx"
OUTPUT_PATH = "mobilenetv2-12-m2.onnx"

def test_m2_mobilenetv2():
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}. Download from ONNX Model Zoo."

    passes = [EliminateDeadNodes(), EliminateIdentityOps()]

    report = optimize(
        model_path=MODEL_PATH,
        output_path=OUTPUT_PATH,
        passes=passes,
        verify_each_pass=True,
        n_verify_samples=10,
    )

    # Node count must not increase
    assert report["nodes_after"] <= report["nodes_before"], \
        f"Node count increased: {report['nodes_before']} → {report['nodes_after']}"

    # Output model must be a valid ONNX graph
    optimized = onnx.load(OUTPUT_PATH)
    onnx.checker.check_model(optimized)

    # Final verify pass
    original = onnx.load(MODEL_PATH)
    vreport = verify(original, optimized, n_samples=10)
    assert vreport.passed, f"Accuracy check failed: max_diff={vreport.max_diff}"

    print(f"\n✓ M2 test passed")
    print(f"  Nodes: {report['nodes_before']} → {report['nodes_after']}")
    print(f"  Size:  {report['size_before_mb']} MB → {report['size_after_mb']} MB")
    print(f"  Max diff: {vreport.max_diff:.2e}")

if __name__ == "__main__":
    test_m2_mobilenetv2()
```

Run it:
```bash
python tests/test_mobilenetv2.py
```

---

## Smoke Test Sequence

Run these in order. Each should pass before moving to the next.

```bash
# 1. Run optimizer with both passes
python optimizer.py mobilenetv2-12.onnx mobilenetv2-12-m2.onnx

# 2. Verify the optimized model against the original
python verify.py mobilenetv2-12.onnx mobilenetv2-12-m2.onnx

# 3. Run the proper test
python tests/test_mobilenetv2.py
```

**Expected output:**
```
Loading: mobilenetv2-12.onnx
  Running pass: eliminate_dead_nodes ... ✓
  Running pass: eliminate_identity_ops ... ✓

Model: mobilenetv2-12.onnx
─────────────────────────────────────────
Nodes before:      105
Nodes after:       ??? (could be same or fewer — MobileNetV2 is already fairly clean)
Size before:       13.32 MB
Size after:        ~13.32 MB
Passes applied:    eliminate_dead_nodes, eliminate_identity_ops
Time:              ~0.5s

✓ Verification passed | max_diff=0.00e+00 | samples=10

✓ M2 test passed
```

> **Note on node count:** MobileNetV2 from the ONNX model zoo is a clean export — you may see zero nodes eliminated. That's expected and correct. These passes have high impact on HuggingFace exports (BERT, Whisper), not clean vision model exports. The passes are working correctly even if the count doesn't change here.

---

## Definition of Done

- [ ] `eliminate_dead_nodes.py` implemented and importable
- [ ] `eliminate_identity_ops.py` implemented and importable
- [ ] `passes/__init__.py` updated
- [ ] `optimizer.py` updated with both passes registered
- [ ] `tests/test_mobilenetv2.py` created
- [ ] `python optimizer.py mobilenetv2-12.onnx mobilenetv2-12-m2.onnx` runs without errors
- [ ] `verify.py` confirms zero accuracy diff on 10 samples
- [ ] `onnx.checker.check_model()` passes on output model
- [ ] Test file runs and prints ✓

---

## Known Gotchas

**Empty output names** — Some nodes have empty strings `""` in their output list (a valid ONNX pattern for optional outputs). The `if out:` guard in `eliminate_dead_nodes` handles this. Don't remove it.

**Identity chain resolution** — `eliminate_identity_ops` resolves chains (Identity → Identity → Identity) via the `resolve()` loop. Without this, rewiring only the immediate next node can leave dangling references. The `visited` set prevents infinite loops on malformed graphs.

**Graph output rewiring** — Don't forget to rewire `graph.output` names, not just node inputs. If an Identity node feeds directly into a graph output, skipping this step produces an invalid graph.

**MobileNetV2 may show 0 eliminated** — This is fine. Run `python -c "import onnx; m=onnx.load('mobilenetv2-12.onnx'); print([n.op_type for n in m.graph.node].count('Identity'))"` to confirm how many Identity nodes exist before running. Clean vision models often have none.

---

## What to Check After Running

```python
# Quick inspection — run in a python shell
import onnx
m = onnx.load("mobilenetv2-12.onnx")
op_counts = {}
for n in m.graph.node:
    op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
print(op_counts)
# Look for 'Identity' count — that's what eliminate_identity_ops will remove
```

This tells you exactly what to expect before you run the pass.

---

## Next: M3

Once M2 passes cleanly → `micro_plan_M3.md` — implement `eliminate_unused_initializers` and `eliminate_duplicate_constants`. These complete Tier 1 and often reduce model **size** more noticeably than node count (since initializers are weights).
