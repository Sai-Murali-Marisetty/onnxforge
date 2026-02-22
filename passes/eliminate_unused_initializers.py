import onnx
from .base_pass import BasePass

class EliminateUnusedInitializers(BasePass):

    @property
    def name(self) -> str:
        return "eliminate_unused_initializers"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        # All names used as inputs across all nodes
        used_inputs = set()
        for node in model.graph.node:
            for inp in node.input:
                if inp:  # skip empty strings (optional inputs)
                    used_inputs.add(inp)

        # Names declared as graph inputs (runtime inputs + legacy weight declarations)
        graph_input_names = {inp.name for inp in model.graph.input}

        # Keep initializers that are actually used
        kept = []
        removed = 0
        for initializer in model.graph.initializer:
            if initializer.name in used_inputs or initializer.name in graph_input_names:
                kept.append(initializer)
            else:
                removed += 1

        # Rebuild initializer list in-place
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)

        if removed > 0:
            print(f"    → removed {removed} unused initializer(s)")

        return model
