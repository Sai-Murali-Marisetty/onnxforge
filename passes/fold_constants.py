import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from .base_pass import BasePass


def _is_constant_node(node):
    return node.op_type == "Constant"


def _collect_constant_names(graph):
    """All tensor names that are statically known constants."""
    constants = set()

    # All initializers are constants
    for init in graph.initializer:
        constants.add(init.name)

    # Explicit Constant op outputs are constants
    for node in graph.node:
        if node.op_type == "Constant":
            constants.update(node.output)

    return constants


def _find_foldable_nodes(graph):
    """
    Propagate forward from known constants.
    A node is foldable if ALL its non-empty inputs are constants.
    Returns (foldable_nodes_list, all_constant_names_set).
    """
    constant_names = _collect_constant_names(graph)
    foldable = []
    visited = set()

    changed = True
    while changed:
        changed = False
        for node in graph.node:
            if id(node) in visited:
                continue
            if node.op_type == "Constant":
                continue
            if all(inp in constant_names or inp == "" for inp in node.input):
                foldable.append(node)
                visited.add(id(node))
                constant_names.update(node.output)
                changed = True

    return foldable, constant_names


def _get_output_names_to_keep(graph, foldable_nodes, constant_names):
    """
    Among constant-valued tensors, find those that are actually consumed
    by non-foldable nodes or are graph outputs.
    These are the ones we need to materialise as Constant nodes.
    """
    graph_output_names = {o.name for o in graph.output}
    
    foldable_output_names = set()
    for node in foldable_nodes:
        foldable_output_names.update(node.output)

    # Find which foldable outputs feed non-foldable nodes
    needed = set()
    foldable_node_ids = {id(n) for n in foldable_nodes}

    for node in graph.node:
        if id(node) in foldable_node_ids:
            continue
        if node.op_type == "Constant":
            continue
        for inp in node.input:
            if inp in foldable_output_names:
                needed.add(inp)

    # Also keep graph outputs that happen to be constant
    needed.update(name for name in foldable_output_names if name in graph_output_names)

    return needed


def _build_mini_model(foldable_nodes, graph, opset_version):
    """
    Build a minimal ONNX model containing only the foldable subgraph.
    Used to run through ORT and get pre-computed values.
    """
    # Collect all initializers referenced by foldable nodes
    init_names_needed = set()
    for node in foldable_nodes:
        for inp in node.input:
            if inp:
                init_names_needed.add(inp)

    # Include initializers needed
    relevant_inits = [
        init for init in graph.initializer
        if init.name in init_names_needed
    ]

    # Also include Constant nodes referenced by foldable nodes
    const_nodes = [
        node for node in graph.node
        if node.op_type == "Constant"
    ]

    all_nodes = const_nodes + foldable_nodes

    # Collect all outputs of foldable nodes as graph outputs
    # Use FLOAT as default type - ORT will infer the actual type
    outputs = []
    for node in foldable_nodes:
        for out in node.output:
            if out:
                outputs.append(
                    helper.make_tensor_value_info(out, TensorProto.FLOAT, None)
                )

    graph_def = helper.make_graph(
        all_nodes,
        "fold_subgraph",
        inputs=[],
        outputs=outputs,
        initializer=relevant_inits,
    )

    opset_imports = [helper.make_opsetid("", opset_version)]
    model = helper.make_model(graph_def, opset_imports=opset_imports)
    model.ir_version = 8
    
    # Let ONNX infer shapes and types
    try:
        from onnx import shape_inference
        model = shape_inference.infer_shapes(model)
    except:
        pass
    
    return model


def _run_subgraph(mini_model):
    """Execute mini model with ORT, return dict of output_name → numpy array."""
    sess = ort.InferenceSession(mini_model.SerializeToString())
    output_names = [o.name for o in mini_model.graph.output]
    results = sess.run(output_names, {})
    return dict(zip(output_names, results))


def _make_constant_node(output_name, array):
    """Create a Constant node pre-loaded with a numpy array."""
    tensor = numpy_helper.from_array(array, name=f"folded_{output_name}")
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[output_name],
        value=tensor,
        name=f"folded_{output_name}",
    )


class FoldConstants(BasePass):

    @property
    def name(self) -> str:
        return "fold_constants"

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        graph = model.graph

        # Detect opset version for building mini model
        opset_version = 13
        for opset in model.opset_import:
            if opset.domain == "" or opset.domain == "ai.onnx":
                opset_version = opset.version

        foldable_nodes, constant_names = _find_foldable_nodes(graph)

        if not foldable_nodes:
            return model

        # Figure out which outputs we actually need to materialise
        needed_outputs = _get_output_names_to_keep(graph, foldable_nodes, constant_names)

        if not needed_outputs:
            # All foldable nodes produce outputs that feed only other foldable nodes
            # which then produce graph outputs. We need to fold everything.
            for node in foldable_nodes:
                for out in node.output:
                    if out in {o.name for o in graph.output}:
                        needed_outputs.add(out)

        if not needed_outputs:
            return model

        # Execute the subgraph
        try:
            mini_model = _build_mini_model(foldable_nodes, graph, opset_version)
            computed = _run_subgraph(mini_model)
        except Exception as e:
            # Folding failed — skip rather than corrupt the model
            print(f"    ⚠ constant folding skipped: {e}")
            return model

        # Build replacement Constant nodes for needed outputs
        new_constant_nodes = []
        for output_name in needed_outputs:
            if output_name in computed:
                new_constant_nodes.append(
                    _make_constant_node(output_name, computed[output_name])
                )

        if not new_constant_nodes:
            return model

        # Remove folded nodes
        foldable_ids = {id(n) for n in foldable_nodes}
        surviving_nodes = [n for n in graph.node if id(n) not in foldable_ids]

        # Also remove Constant nodes whose outputs are now replaced
        replaced_outputs = {n.output[0] for n in new_constant_nodes}
        const_outputs_folded = set()
        for n in foldable_nodes:
            const_outputs_folded.update(n.input)
        
        surviving_nodes = [
            n for n in surviving_nodes
            if not (n.op_type == "Constant" and 
                    all(out in const_outputs_folded for out in n.output))
        ]

        # Add new pre-computed Constant nodes
        surviving_nodes.extend(new_constant_nodes)

        del graph.node[:]
        graph.node.extend(surviving_nodes)

        folded_count = len(foldable_nodes)
        replacement_count = len(new_constant_nodes)
        print(f"    → folded {folded_count} node(s) into {replacement_count} constant(s)")

        return model
