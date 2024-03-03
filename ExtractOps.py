import onnx
import sys

def list_onnx_operators(filename: str) -> None:
   model_proto = onnx.load(filename)
   operators = set()
   count = 0;
   # Traverse each node in the graph
   for node in model_proto.graph.node:
      operators.add(node.op_type)
      count = count+1

   print(operators);
   print("Number of Ops:" + str(count))

   for ten_proto in model_proto.graph.initializer:
     print(ten_proto.name);

   sys.exit();

# Example usage
onnx_file_path = sys.argv[1]
list_onnx_operators(onnx_file_path)