import onnx
from onnx import helper
from .base_pass import BasePass


def _find_relu_consumer(graph, conv_output):
    """Find a Relu node that directly consumes conv_output. Returns node or None."""
    for node in graph.node:
        if node.op_type == "Relu" and node.input[0] == conv_output:
            return node
    return None


def _count_consumers(graph, tensor_name):
    """Count how many nodes consume a tensor."""
    count = 0
    for node in graph.node:
        if tensor_name in node.input:
            count += 1
    return count


class FuseConvRelu(BasePass):
    """
    Identify Conv+Relu patterns for potential fusion by downstream converters.
    
    This pass finds Conv nodes followed by Relu and records them for potential
    optimization. In standard ONNX, Conv and Relu remain separate ops.
    
    For TFLite export, converters like tf2onnx can use this pattern information
    to create fused Conv2D ops with activation='RELU'.
    
    This pass currently operates as a pattern detector. Full fusion with
    attribute annotation is disabled because ONNX Runtime doesn't support
    custom 'activation' attributes on Conv.
    
    The pass still provides value by:
    1. Logging how many Conv+Relu pairs exist (optimization opportunity)
    2. Preparing for future TFLite-specific export mode
    """

    @property
    def name(self) -> str:
        return "fuse_conv_relu"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        graph = model.graph
        graph_output_names = {o.name for o in graph.output}

        conv_relu_pairs = 0

        for node in graph.node:
            if node.op_type != "Conv":
                continue

            conv_output = node.output[0]

            if conv_output in graph_output_names:
                continue

            relu_node = _find_relu_consumer(graph, conv_output)
            if relu_node is None:
                continue

            # Only count if Conv output is consumed ONLY by this Relu
            if _count_consumers(graph, conv_output) == 1:
                conv_relu_pairs += 1

        if conv_relu_pairs > 0:
            print(f"    → found {conv_relu_pairs} Conv+ReLU pair(s) (pattern detected, not fused for ORT compatibility)")

        return model
