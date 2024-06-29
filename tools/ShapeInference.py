import onnx
from onnx import shape_inference
path = "D:\Projects\RapidWebNN\RapidChat\RapidChat\RapidChat\models\onnx\decoder_model_merged_quantized.onnx"
onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), "D:\Projects\RapidWebNN\RapidChat\RapidChat\RapidChat\models\onnx\decoder_model_merged_quantized_si.onnx")