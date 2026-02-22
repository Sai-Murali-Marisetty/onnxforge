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
