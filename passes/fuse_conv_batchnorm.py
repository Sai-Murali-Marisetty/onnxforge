import onnx
import numpy as np
from onnx import numpy_helper
from .base_pass import BasePass


def _get_initializer(graph, name):
    """Fetch a named initializer as a numpy array. Returns None if not found."""
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def _set_initializer(graph, name, array):
    """Update or create a named initializer from a numpy array."""
    tensor = numpy_helper.from_array(array.astype(np.float32), name=name)
    for i, init in enumerate(graph.initializer):
        if init.name == name:
            graph.initializer[i].CopyFrom(tensor)
            return
    graph.initializer.append(tensor)


def _find_bn_consumer(graph, conv_output):
    """
    Find a BatchNormalization node that directly consumes the given tensor name.
    Returns the BN node or None.
    """
    for node in graph.node:
        if node.op_type == "BatchNormalization":
            if node.input[0] == conv_output:
                return node
    return None


def _fuse_conv_bn(conv_node, bn_node, graph):
    """
    Fold BN into Conv. Returns True if fusion succeeded, False otherwise.
    Modifies graph in-place.
    """
    # Conv inputs: [X, W] or [X, W, B]
    weight_name = conv_node.input[1]
    bias_name   = conv_node.input[2] if len(conv_node.input) > 2 else None

    # BN inputs: [X, scale(gamma), bias(beta), mean, var]
    gamma_name = bn_node.input[1]
    beta_name  = bn_node.input[2]
    mean_name  = bn_node.input[3]
    var_name   = bn_node.input[4]

    # Load all arrays
    weight = _get_initializer(graph, weight_name)
    gamma  = _get_initializer(graph, gamma_name)
    beta   = _get_initializer(graph, beta_name)
    mean   = _get_initializer(graph, mean_name)
    var    = _get_initializer(graph, var_name)

    if any(x is None for x in [weight, gamma, beta, mean, var]):
        return False  # dynamic BN params — cannot fuse statically

    bias = _get_initializer(graph, bias_name) if bias_name else np.zeros(weight.shape[0], dtype=np.float32)

    # Extract epsilon from BN attributes
    eps = 1e-5
    for attr in bn_node.attribute:
        if attr.name == "epsilon":
            eps = attr.f
            break

    # Compute fused weights
    scale      = gamma / np.sqrt(var + eps)         # [out_channels]
    
    # Handle different weight tensor shapes (regular conv vs depthwise)
    if len(weight.shape) == 4:
        scale_shape = (-1, 1, 1, 1)
    elif len(weight.shape) == 3:
        scale_shape = (-1, 1, 1)
    elif len(weight.shape) == 2:
        scale_shape = (-1, 1)
    else:
        scale_shape = (-1,)
    
    scale_broadcast = scale.reshape(scale_shape)
    new_weight = weight * scale_broadcast
    new_bias   = (bias - mean) * scale + beta        # [out_channels]

    # Update Conv weight initializer in-place
    _set_initializer(graph, weight_name, new_weight)

    # Set Conv bias
    fused_bias_name = bias_name if bias_name else f"{weight_name}_fused_bias"
    _set_initializer(graph, fused_bias_name, new_bias)

    # Update Conv node inputs to include bias if it didn't before
    if not bias_name:
        conv_node.input.append(fused_bias_name)

    # Rewire Conv output → BN output (skip BN entirely)
    bn_output = bn_node.output[0]
    conv_node.output[0] = bn_output

    return True


class FuseConvBatchnorm(BasePass):

    @property
    def name(self) -> str:
        return "fuse_conv_batchnorm"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        graph = model.graph

        # Two-pass approach: first identify all pairs, then apply fusions
        # This avoids issues with modifying the graph while iterating.

        fusion_pairs = []  # List of (conv_node, bn_node)

        for node in graph.node:
            if node.op_type != "Conv":
                continue

            conv_output = node.output[0]
            bn_node = _find_bn_consumer(graph, conv_output)
            if bn_node is not None:
                fusion_pairs.append((node, bn_node))

        if not fusion_pairs:
            return model

        # Collect BN nodes to remove (by their original ID before modification)
        bn_nodes_to_remove = set()

        for conv_node, bn_node in fusion_pairs:
            success = _fuse_conv_bn(conv_node, bn_node, graph)
            if success:
                bn_nodes_to_remove.add(id(bn_node))

        # Remove all fused BN nodes at once
        if bn_nodes_to_remove:
            new_nodes = [n for n in graph.node if id(n) not in bn_nodes_to_remove]
            del graph.node[:]
            graph.node.extend(new_nodes)
            print(f"    → fused {len(bn_nodes_to_remove)} Conv+BN pair(s)")

        return model
