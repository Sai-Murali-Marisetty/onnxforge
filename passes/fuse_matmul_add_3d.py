"""
Pass 12: Fold weight transposes for 3D MatMul patterns in transformers.

HuggingFace exports linear layers as:
    Transpose(Weight:[H', H], perm=[1,0]) -> Weight_T:[H, H']
    MatMul(X:[B,S,H], Weight_T) -> Y

This pass folds the transpose into the weight at optimization time:
- Transpose the weight array in the initializer
- Remove the Transpose node
- Connect MatMul directly to the transposed weight

Net reduction: 1 node per linear layer (removes Transpose).
For BERT-base with 72 linear projections, this is -72 nodes.

Note: The MatMul+Add fusion to Gemm is tricky because Gemm requires 2D inputs.
For 3D matmuls, we keep them as MatMul+Add since ORT handles this efficiently.
"""

import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto, shape_inference
from .base_pass import BasePass
from collections import defaultdict


def _get_initializer_array(graph, name):
    """Get initializer as numpy array."""
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def _is_initializer(graph, name):
    """True if name is an initializer."""
    return any(init.name == name for init in graph.initializer)


def _build_consumer_map(graph):
    """Map output name -> list of consumer nodes."""
    result = defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            result[inp].append(node)
    return result


class FuseMatmulAdd3d(BasePass):
    """
    Fold weight transposes in transformer linear layers.
    
    Pattern:
        Transpose(W:[H',H], perm=[1,0]) -> W_T:[H,H']
        MatMul(X:[B,S,H], W_T) -> Y
    
    Replacement:
        W_transposed as new initializer
        MatMul(X, W_transposed) -> Y
    
    This removes the Transpose node by pre-computing the transpose.
    """

    @property
    def name(self) -> str:
        return "fuse_matmul_add_3d"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        graph = model.graph
        consumer_map = _build_consumer_map(graph)
        
        nodes_to_remove = set()
        new_initializers = {}  # old_name -> new transposed initializer
        input_rewrites = {}    # old output name -> new initializer name
        
        # Find Transpose nodes that:
        # 1. Transpose an initializer
        # 2. Use perm=[1, 0]
        # 3. Output goes only to MatMul
        for node in graph.node:
            if node.op_type != "Transpose":
                continue
            
            if len(node.input) != 1:
                continue
            
            weight_name = node.input[0]
            if not _is_initializer(graph, weight_name):
                continue
            
            # Check perm attribute
            perm = None
            for attr in node.attribute:
                if attr.name == "perm":
                    perm = list(attr.ints)
                    break
            
            if perm != [1, 0]:
                continue
            
            # Check that output only goes to MatMul
            transpose_output = node.output[0]
            consumers = consumer_map.get(transpose_output, [])
            
            if not all(c.op_type == "MatMul" for c in consumers):
                continue
            
            if len(consumers) == 0:
                continue
            
            # Get weight array and transpose it
            weight_array = _get_initializer_array(graph, weight_name)
            if weight_array is None or weight_array.ndim != 2:
                continue
            
            # Create transposed weight
            transposed = weight_array.T.copy()  # Actually transpose
            new_name = f"{weight_name}_transposed"
            
            new_initializers[weight_name] = numpy_helper.from_array(transposed, new_name)
            input_rewrites[transpose_output] = new_name
            nodes_to_remove.add(node.name)
        
        if not nodes_to_remove:
            return model
        
        # Build new node list with input rewrites
        final_nodes = []
        for node in graph.node:
            if node.name in nodes_to_remove:
                continue
            
            # Check if any inputs need rewriting
            new_inputs = []
            for inp in node.input:
                if inp in input_rewrites:
                    new_inputs.append(input_rewrites[inp])
                else:
                    new_inputs.append(inp)
            
            if new_inputs != list(node.input):
                # Create new node with rewritten inputs
                new_node = helper.make_node(
                    node.op_type,
                    inputs=new_inputs,
                    outputs=list(node.output),
                    name=node.name
                )
                # Copy attributes
                new_node.attribute.extend(node.attribute)
                final_nodes.append(new_node)
            else:
                final_nodes.append(node)
        
        # Build new initializer list
        final_initializers = list(graph.initializer)
        for init in new_initializers.values():
            final_initializers.append(init)
        
        # Create new graph
        new_graph = helper.make_graph(
            final_nodes,
            graph.name,
            graph.input,
            graph.output,
            final_initializers
        )
        new_graph.value_info.extend(graph.value_info)
        
        # Create new model
        new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
        new_model.ir_version = model.ir_version
        
        return new_model
