import onnx
from onnx import helper, numpy_helper, TensorProto
from .base_pass import BasePass


def _build_output_to_node(graph):
    """Map: output_name → node that produces it."""
    return {out: node for node in graph.node for out in node.output}


def _build_input_consumers(graph):
    """Map: tensor_name → list of nodes that consume it."""
    consumers = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    return consumers


def _get_constant_value(name, graph, output_to_node):
    """
    Try to get the numpy value of a named tensor if it's a constant.
    Checks initializers first, then Constant nodes.
    Returns numpy array or None.
    """
    # Check initializers
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)

    # Check Constant nodes
    if name in output_to_node:
        node = output_to_node[name]
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    return numpy_helper.to_array(attr.t)

    return None


def _get_static_shape(name, graph):
    """
    Try to get the static shape of a tensor from graph value_info or graph inputs.
    Returns list of ints or None if shape is dynamic/unknown.
    """
    # Check graph inputs
    for inp in graph.input:
        if inp.name == name:
            dims = inp.type.tensor_type.shape.dim
            shape = []
            for d in dims:
                if d.dim_value > 0:
                    shape.append(d.dim_value)
                else:
                    return None  # dynamic dim
            return shape

    # Check value_info (shape inference results)
    for vi in graph.value_info:
        if vi.name == name:
            dims = vi.type.tensor_type.shape.dim
            shape = []
            for d in dims:
                if d.dim_value > 0:
                    shape.append(d.dim_value)
                else:
                    return None
            return shape

    return None


class SimplifyShapeChains(BasePass):

    @property
    def name(self) -> str:
        return "simplify_shape_chains"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        # Run shape inference first so value_info is populated
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception:
            pass  # shape inference can fail on unusual models — proceed anyway

        graph = model.graph
        output_to_node = _build_output_to_node(graph)
        consumers = _build_input_consumers(graph)
        graph_output_names = {o.name for o in graph.output}

        nodes_to_remove = set()
        rewire = {}
        removed_count = 0

        for node in graph.node:
            if id(node) in nodes_to_remove:
                continue

            # --- Pattern: Redundant Reshape (identity reshape) ---
            if node.op_type == "Reshape":
                input_name  = node.input[0]
                shape_input = node.input[1] if len(node.input) > 1 else None
                output_name = node.output[0]

                if shape_input:
                    shape_val = _get_constant_value(shape_input, graph, output_to_node)
                    input_shape = _get_static_shape(input_name, graph)

                    if shape_val is not None and input_shape is not None:
                        target_shape = shape_val.flatten().tolist()
                        # Check if reshape is identity (same shape, no -1 dims)
                        if (len(target_shape) == len(input_shape) and
                                all(int(t) == s for t, s in zip(target_shape, input_shape)) and
                                -1 not in target_shape):
                            # This Reshape does nothing — remove it
                            rewire[output_name] = input_name
                            nodes_to_remove.add(id(node))
                            removed_count += 1

            # --- Pattern: Shape node whose output is fully constant (dead after folding) ---
            if node.op_type == "Shape":
                output_name = node.output[0]
                # If this Shape output only feeds nodes that are being removed or
                # is never consumed, mark it dead
                output_consumers = consumers.get(output_name, [])
                if not output_consumers and output_name not in graph_output_names:
                    nodes_to_remove.add(id(node))
                    removed_count += 1

        if not nodes_to_remove and not rewire:
            return model

        # Apply rewiring
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
        del graph.node[:]
        graph.node.extend(new_nodes)

        if removed_count > 0:
            print(f"    → simplified {removed_count} shape chain node(s)")

        return model
