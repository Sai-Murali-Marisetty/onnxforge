import onnx
from onnx import helper
from .base_pass import BasePass


def _get_perm(node):
    """Extract perm attribute from a Transpose node as a list of ints."""
    for attr in node.attribute:
        if attr.name == "perm":
            return list(attr.ints)
    return None


def _compose_perms(p1, p2):
    """Compose two permutations: apply p1 first, then p2."""
    return [p1[p2[i]] for i in range(len(p2))]


def _is_identity_perm(perm):
    return perm == list(range(len(perm)))


def _make_transpose_node(input_name, output_name, perm, name):
    return helper.make_node(
        "Transpose",
        inputs=[input_name],
        outputs=[output_name],
        perm=perm,
        name=name,
    )


class EliminateRedundantTransposes(BasePass):

    @property
    def name(self) -> str:
        return "eliminate_redundant_transposes"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        changed = True
        total_removed = 0

        while changed:
            model, removed = self._run_one_pass(model)
            total_removed += removed
            changed = removed > 0

        if total_removed > 0:
            print(f"    → eliminated {total_removed} redundant transpose(s)")

        return model

    def _run_one_pass(self, model: onnx.ModelProto):
        graph = model.graph
        nodes = list(graph.node)

        # Map: output_name → node that produces it
        output_to_node = {}
        for node in nodes:
            for out in node.output:
                output_to_node[out] = node

        # Map: input_name → list of nodes that consume it
        input_to_consumers = {}
        for node in nodes:
            for inp in node.input:
                if inp:
                    if inp not in input_to_consumers:
                        input_to_consumers[inp] = []
                    input_to_consumers[inp].append(node)

        # Graph output names — cannot remove nodes whose output is a graph output
        graph_output_names = {o.name for o in graph.output}

        nodes_to_remove = set()
        nodes_to_add = []
        rewire = {}  # old_output → new_output (for downstream rewiring)

        for node in nodes:
            if node.op_type != "Transpose":
                continue
            if id(node) in nodes_to_remove:
                continue

            # Does this Transpose feed into another Transpose?
            out = node.output[0]
            consumers = input_to_consumers.get(out, [])
            
            # Only optimize if there's exactly one consumer and it's a Transpose
            if len(consumers) != 1:
                continue
            next_node = consumers[0]
            if next_node.op_type != "Transpose":
                continue
            if id(next_node) in nodes_to_remove:
                continue

            # Don't collapse if intermediate output is a graph output
            if out in graph_output_names:
                continue

            p1 = _get_perm(node)
            p2 = _get_perm(next_node)

            if p1 is None or p2 is None:
                continue

            composed = _compose_perms(p1, p2)

            if _is_identity_perm(composed):
                # Remove both — rewire input of first directly to output of second
                rewire[next_node.output[0]] = node.input[0]
                nodes_to_remove.add(id(node))
                nodes_to_remove.add(id(next_node))

            else:
                # Replace both with one composed Transpose
                new_node = _make_transpose_node(
                    input_name=node.input[0],
                    output_name=next_node.output[0],
                    perm=composed,
                    name=f"{node.name}_fused",
                )
                nodes_to_remove.add(id(node))
                nodes_to_remove.add(id(next_node))
                nodes_to_add.append(new_node)

        if not nodes_to_remove:
            return model, 0

        # Apply rewiring to all node inputs
        for n in graph.node:
            for i, inp in enumerate(n.input):
                if inp in rewire:
                    n.input[i] = rewire[inp]

        # Apply rewiring to graph outputs
        for out in graph.output:
            if out.name in rewire:
                out.name = rewire[out.name]

        # Rebuild node list
        new_nodes = [n for n in graph.node if id(n) not in nodes_to_remove]
        new_nodes.extend(nodes_to_add)

        del graph.node[:]
        graph.node.extend(new_nodes)

        return model, len(nodes_to_remove)
