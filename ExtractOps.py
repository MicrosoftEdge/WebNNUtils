import onnx

def list_onnx_operators(filename: str) -> None:
   model_proto = onnx.load(filename)
   operators = set()
   # Traverse each node in the graph
   for node in model_proto.graph.node:
      operators.add(node.op_type)

   print(operators);

# Example usage
onnx_file_path = "D:\\Projects\\TinyLlama-Chat-v1.1-onnx_quantized\\onnx\\decoder_model_merged_quantized.onnx"
list_onnx_operators(onnx_file_path)