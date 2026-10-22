"""
Tests for ONNX exported model and ONNX Runtime execution.
"""
from pathlib import Path
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path("D:/auto_drive")
ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "best.onnx"

def test_onnx_model_file_exists():
    assert ONNX_MODEL_PATH.exists(), "ONNX model file must exist in models/best.onnx"
    assert ONNX_MODEL_PATH.stat().st_size > 1024 * 1024, "ONNX model must be > 1MB"

def test_onnx_runtime_session_execution():
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    
    assert len(inputs) == 1
    assert inputs[0].name == "images"
    # Expected shape: [1, 3, 640, 640]
    dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
    
    out = session.run([outputs[0].name], {inputs[0].name: dummy_input})
    assert len(out) == 1
    # Expected output shape: [1, 14, 8400]
    assert out[0].shape == (1, 14, 8400)
