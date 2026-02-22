"""Tests for fuse_matmul_add_3d pass (Pass 12)."""
import pytest
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from passes.fuse_matmul_add_3d import FuseMatmulAdd3d


def _make_transformer_linear_model():
    """
    Create a model with HuggingFace-style linear layer:
    Transpose(weight) -> MatMul(input, transposed_weight) -> Add(bias)
    """
    # Weight: [out_features, in_features] = [64, 128]
    weight = np.random.randn(64, 128).astype(np.float32)
    bias = np.random.randn(64).astype(np.float32)
    
    weight_init = numpy_helper.from_array(weight, "weight")
    bias_init = numpy_helper.from_array(bias, "bias")
    
    # Transpose: [64, 128] -> [128, 64]
    transpose_node = helper.make_node(
        "Transpose",
        inputs=["weight"],
        outputs=["weight_t"],
        perm=[1, 0],
        name="transpose"
    )
    
    # MatMul: [B, S, 128] @ [128, 64] -> [B, S, 64]
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["input", "weight_t"],
        outputs=["matmul_out"],
        name="matmul"
    )
    
    # Add: [B, S, 64] + [64] -> [B, S, 64]
    add_node = helper.make_node(
        "Add",
        inputs=["matmul_out", "bias"],
        outputs=["output"],
        name="add"
    )
    
    graph = helper.make_graph(
        [transpose_node, matmul_node, add_node],
        "transformer_linear",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 128])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 64])],
        [weight_init, bias_init]
    )
    
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def _make_two_linear_layers():
    """Create a model with two consecutive HF-style linear layers."""
    # Layer 1: 128 -> 256
    w1 = np.random.randn(256, 128).astype(np.float32)
    b1 = np.random.randn(256).astype(np.float32)
    
    # Layer 2: 256 -> 64
    w2 = np.random.randn(64, 256).astype(np.float32)
    b2 = np.random.randn(64).astype(np.float32)
    
    nodes = [
        # Layer 1
        helper.make_node("Transpose", ["w1"], ["w1_t"], perm=[1, 0], name="t1"),
        helper.make_node("MatMul", ["input", "w1_t"], ["mm1"], name="mm1"),
        helper.make_node("Add", ["mm1", "b1"], ["out1"], name="add1"),
        # Layer 2
        helper.make_node("Transpose", ["w2"], ["w2_t"], perm=[1, 0], name="t2"),
        helper.make_node("MatMul", ["out1", "w2_t"], ["mm2"], name="mm2"),
        helper.make_node("Add", ["mm2", "b2"], ["output"], name="add2"),
    ]
    
    graph = helper.make_graph(
        nodes,
        "two_layers",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 128])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 64])],
        [
            numpy_helper.from_array(w1, "w1"),
            numpy_helper.from_array(b1, "b1"),
            numpy_helper.from_array(w2, "w2"),
            numpy_helper.from_array(b2, "b2"),
        ]
    )
    
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def _make_no_transpose_model():
    """Create a model where weight is already in correct orientation."""
    weight = np.random.randn(128, 64).astype(np.float32)  # Already [in, out]
    bias = np.random.randn(64).astype(np.float32)
    
    nodes = [
        helper.make_node("MatMul", ["input", "weight"], ["mm"], name="mm"),
        helper.make_node("Add", ["mm", "bias"], ["output"], name="add"),
    ]
    
    graph = helper.make_graph(
        nodes,
        "no_transpose",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 128])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 64])],
        [
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(bias, "bias"),
        ]
    )
    
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


class TestFuseMatmulAdd3d:
    """Test suite for the transpose folding pass."""
    
    def test_single_linear_layer(self):
        """Test folding a single HF-style linear layer."""
        model = _make_transformer_linear_model()
        assert len(model.graph.node) == 3  # Transpose + MatMul + Add
        
        fuse = FuseMatmulAdd3d()
        optimized = fuse.run(model)
        
        # Should remove Transpose
        assert len(optimized.graph.node) == 2  # MatMul + Add only
        
        # Check no Transpose nodes remain
        op_types = [n.op_type for n in optimized.graph.node]
        assert "Transpose" not in op_types
        assert "MatMul" in op_types
        assert "Add" in op_types
        
        # Should have added transposed weight initializer
        init_names = [i.name for i in optimized.graph.initializer]
        assert any("transposed" in name for name in init_names)
    
    def test_two_linear_layers(self):
        """Test folding two consecutive linear layers."""
        model = _make_two_linear_layers()
        assert len(model.graph.node) == 6  # 2 * (Transpose + MatMul + Add)
        
        fuse = FuseMatmulAdd3d()
        optimized = fuse.run(model)
        
        # Should remove both Transposes
        assert len(optimized.graph.node) == 4  # 2 * (MatMul + Add)
        
        op_types = [n.op_type for n in optimized.graph.node]
        assert op_types.count("Transpose") == 0
        assert op_types.count("MatMul") == 2
        assert op_types.count("Add") == 2
    
    def test_no_change_without_transpose(self):
        """Test that pass doesn't modify models without the pattern."""
        model = _make_no_transpose_model()
        original_nodes = len(model.graph.node)
        
        fuse = FuseMatmulAdd3d()
        optimized = fuse.run(model)
        
        # Should not change
        assert len(optimized.graph.node) == original_nodes
    
    def test_accuracy_preserved(self):
        """Test that optimization preserves numerical accuracy."""
        import onnxruntime as ort
        
        model = _make_transformer_linear_model()
        fuse = FuseMatmulAdd3d()
        optimized = fuse.run(model)
        
        # Run both models
        sess_orig = ort.InferenceSession(model.SerializeToString())
        sess_opt = ort.InferenceSession(optimized.SerializeToString())
        
        # Random input
        test_input = np.random.randn(1, 16, 128).astype(np.float32)
        
        orig_out = sess_orig.run(None, {"input": test_input})[0]
        opt_out = sess_opt.run(None, {"input": test_input})[0]
        
        # Should be identical
        np.testing.assert_allclose(orig_out, opt_out, rtol=1e-5, atol=1e-5)


# Standalone test
def test_single_linear_layer():
    TestFuseMatmulAdd3d().test_single_linear_layer()

def test_two_linear_layers():
    TestFuseMatmulAdd3d().test_two_linear_layers()

def test_no_change():
    TestFuseMatmulAdd3d().test_no_change_without_transpose()

def test_accuracy():
    TestFuseMatmulAdd3d().test_accuracy_preserved()
