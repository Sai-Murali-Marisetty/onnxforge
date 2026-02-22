import onnx
import hashlib
from onnx import numpy_helper
from .base_pass import BasePass

def _hash_initializer(tensor) -> str:
    """Hash an initializer by its dtype, shape, and raw data."""
    arr = numpy_helper.to_array(tensor)
    meta = f"{arr.dtype}:{arr.shape}"
    data_hash = hashlib.md5(arr.tobytes()).hexdigest()
    return f"{meta}:{data_hash}"

class EliminateDuplicateConstants(BasePass):

    @property
    def name(self) -> str:
        return "eliminate_duplicate_constants"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        # Group initializers by content hash
        hash_to_canonical = {}   # hash → first initializer name seen
        remap = {}               # duplicate name → canonical name

        for initializer in model.graph.initializer:
            h = _hash_initializer(initializer)
            if h not in hash_to_canonical:
                hash_to_canonical[h] = initializer.name
            else:
                # This initializer is a duplicate — map it to the canonical one
                canonical = hash_to_canonical[h]
                if initializer.name != canonical:
                    remap[initializer.name] = canonical

        if not remap:
            return model  # nothing to do

        # Rewrite all node inputs
        for node in model.graph.node:
            for i, inp in enumerate(node.input):
                if inp in remap:
                    node.input[i] = remap[inp]

        # Rewrite graph outputs (rare but correct)
        for output in model.graph.output:
            if output.name in remap:
                output.name = remap[output.name]

        # Remove duplicate initializers
        kept = [init for init in model.graph.initializer
                if init.name not in remap]

        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)

        print(f"    → removed {len(remap)} duplicate constant(s)")

        return model
