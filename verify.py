import numpy as np
import onnxruntime as ort
import onnx
from dataclasses import dataclass
from typing import Optional

TOLERANCE = 1e-5  # max acceptable absolute difference

@dataclass
class VerificationReport:
    passed: bool
    max_diff: float
    samples: int
    failed_at_sample: Optional[int] = None

class AccuracyLossError(Exception):
    pass

def _get_input_specs(model: onnx.ModelProto):
    """Extract input names and shapes from model graph."""
    inputs = []
    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            # Use 1 for dynamic/unknown dims
            shape.append(dim.dim_value if dim.dim_value > 0 else 1)
        dtype = inp.type.tensor_type.elem_type
        inputs.append((inp.name, shape, dtype))
    return inputs

def _generate_random_input(input_specs):
    """Generate random numpy arrays matching model input specs."""
    feed = {}
    for name, shape, dtype in input_specs:
        if dtype == 1:   # FLOAT
            feed[name] = np.random.randn(*shape).astype(np.float32)
        elif dtype == 7: # INT64
            # For token_type_ids in BERT/RoBERTa models, use 0 only (safest)
            # For other int64 inputs (input_ids, attention_mask), use small range
            if 'token_type' in name:
                feed[name] = np.zeros(shape, dtype=np.int64)
            elif 'attention_mask' in name:
                feed[name] = np.ones(shape, dtype=np.int64)
            else:
                feed[name] = np.random.randint(0, 100, shape).astype(np.int64)
        elif dtype == 6: # INT32
            feed[name] = np.random.randint(0, 100, shape).astype(np.int32)
        else:
            feed[name] = np.random.randn(*shape).astype(np.float32)
    return feed

def verify(
    original_model: onnx.ModelProto,
    optimized_model: onnx.ModelProto,
    n_samples: int = 5,
    tolerance: float = TOLERANCE
) -> VerificationReport:
    """
    Run n_samples random inputs through both models.
    Compare all outputs. Raise AccuracyLossError if diff exceeds tolerance.
    """
    sess_orig = ort.InferenceSession(original_model.SerializeToString())
    sess_opt  = ort.InferenceSession(optimized_model.SerializeToString())

    input_specs = _get_input_specs(original_model)
    
    # Get output names separately - they may differ after optimization
    output_names_orig = [o.name for o in original_model.graph.output]
    output_names_opt = [o.name for o in optimized_model.graph.output]

    global_max_diff = 0.0

    for i in range(n_samples):
        feed = _generate_random_input(input_specs)

        out_orig = sess_orig.run(output_names_orig, feed)
        out_opt  = sess_opt.run(output_names_opt, feed)

        for orig, opt in zip(out_orig, out_opt):
            diff = np.max(np.abs(np.array(orig) - np.array(opt)))
            global_max_diff = max(global_max_diff, diff)

            if diff > tolerance:
                raise AccuracyLossError(
                    f"Sample {i}: max diff {diff:.2e} exceeds tolerance {tolerance:.2e}"
                )

    return VerificationReport(passed=True, max_diff=global_max_diff, samples=n_samples)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python verify.py original.onnx optimized.onnx")
        sys.exit(1)

    orig = onnx.load(sys.argv[1])
    opt  = onnx.load(sys.argv[2])
    report = verify(orig, opt)
    print(f"✓ Verification passed | max_diff={report.max_diff:.2e} | samples={report.samples}")
