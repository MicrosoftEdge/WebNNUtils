import onnx
from onnx import shape_inference
path = "Test.onnx"
onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), "Test_si.onnx")