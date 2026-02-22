import onnx
import numpy as np
from onnx import helper, numpy_helper
from .base_pass import BasePass


def _get_initializer_array(graph, name):
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def _get_input_rank(graph, value_info_dict, name):
    """
    Get the rank (number of dimensions) of a tensor by name.
    Returns None if unknown.
    """
    # Check value_info first
    if name in value_info_dict:
        shape = value_info_dict[name]
        if shape is not None:
            return len(shape)
    
    # Check graph inputs
    for inp in graph.input:
        if inp.name == name:
            dims = inp.type.tensor_type.shape.dim
            if dims:
                return len(dims)
    
    # Check initializers (weights are always known)
    for init in graph.initializer:
        if init.name == name:
            return len(init.dims)
    
    return None


def _build_value_info_dict(graph):
    """Build a dictionary mapping tensor names to their shapes."""
    result = {}
    for vi in graph.value_info:
        dims = vi.type.tensor_type.shape.dim
        if dims:
            shape = []
            for d in dims:
                if d.dim_value > 0:
                    shape.append(d.dim_value)
                else:
                    shape.append(None)  # dynamic dim
            result[vi.name] = shape
        else:
            result[vi.name] = None
    return result


def _is_constant(graph, name):
    """True if name is an initializer or Constant node output."""
    for init in graph.initializer:
        if init.name == name:
            return True
    for node in graph.node:
        if node.op_type == "Constant" and name in node.output:
            return True
    return False


def _find_add_consumer(graph, matmul_output):
    """Find an Add node that consumes matmul_output as one of its inputs."""
    for node in graph.node:
        if node.op_type == "Add":
            if matmul_output in node.input:
                return node
    return None


def _get_bias_input(add_node, matmul_output):
    """
    Given an Add node and the MatMul output name,
    return the name of the bias input (the other input to Add).
    """
    for inp in add_node.input:
        if inp != matmul_output:
            return inp
    return None


class FuseMatmulAdd(BasePass):

    @property
    def name(self) -> str:
        return "fuse_matmul_add"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        graph = model.graph
        graph_output_names = {o.name for o in graph.output}

        # Build value_info dict for shape checking
        value_info_dict = _build_value_info_dict(graph)

        # Two-pass approach:
        # 1. Identify all MatMul+Add pairs to fuse
        # 2. Build new node list with Gemm in place of MatMul, Add removed

        fusion_pairs = []  # List of (matmul_node, add_node, gemm_node)

        for node in graph.node:
            if node.op_type != "MatMul":
                continue

            matmul_output = node.output[0]

            if matmul_output in graph_output_names:
                continue

            # Gemm requires rank-2 inputs. Skip 3D+ MatMuls (e.g., attention).
            input_a_rank = _get_input_rank(graph, value_info_dict, node.input[0])
            input_b_rank = _get_input_rank(graph, value_info_dict, node.input[1])
            
            # Only fuse if BOTH inputs are known to be rank-2
            if input_a_rank is None or input_b_rank is None:
                # Can't determine rank - skip to be safe
                continue
            if input_a_rank != 2 or input_b_rank != 2:
                # Not a rank-2 MatMul - Gemm won't work
                continue

            add_node = _find_add_consumer(graph, matmul_output)
            if add_node is None:
                continue

            bias_name = _get_bias_input(add_node, matmul_output)
            if bias_name is None:
                continue

            # Bias must be a static constant
            if not _is_constant(graph, bias_name):
                continue

            # Bias should be 1D (linear layer bias)
            bias_array = _get_initializer_array(graph, bias_name)
            if bias_array is not None and bias_array.ndim != 1:
                continue  # skip non-standard bias shapes

            # Build replacement Gemm node
            gemm_node = helper.make_node(
                "Gemm",
                inputs=[node.input[0], node.input[1], bias_name],
                outputs=[add_node.output[0]],
                alpha=1.0,
                beta=1.0,
                transB=0,
                name=f"{node.name}_gemm" if node.name else "fused_gemm",
            )

            fusion_pairs.append((node, add_node, gemm_node))

        if not fusion_pairs:
            return model

        # Build sets of nodes to replace
        matmul_ids = {id(pair[0]) for pair in fusion_pairs}
        add_ids    = {id(pair[1]) for pair in fusion_pairs}
        matmul_to_gemm = {id(pair[0]): pair[2] for pair in fusion_pairs}

        # Rebuild node list: replace MatMul with Gemm, skip Add
        new_nodes = []
        for node in graph.node:
            node_id = id(node)
            if node_id in matmul_ids:
                # Replace MatMul with Gemm (at same position for topological order)
                new_nodes.append(matmul_to_gemm[node_id])
            elif node_id in add_ids:
                # Skip Add node (fused into Gemm)
                pass
            else:
                new_nodes.append(node)

        del graph.node[:]
        graph.node.extend(new_nodes)

        print(f"    → fused {len(fusion_pairs)} MatMul+Add → Gemm")

        return model
