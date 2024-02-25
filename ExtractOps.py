import onnx
import sys

def list_onnx_operators(filename: str) -> None:
   model_proto = onnx.load(filename)
   operators = set()
   # Traverse each node in the graph
   for node in model_proto.graph.node:
      operators.add(node.op_type)

   print(operators);

# Example usage
onnx_file_path = sys.argv[1]
list_onnx_operators(onnx_file_path)