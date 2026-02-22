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
