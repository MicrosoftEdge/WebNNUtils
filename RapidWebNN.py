import onnx
from onnx import helper
from onnx import TensorProto
import onnx.numpy_helper
import numpy as np
import sys
import argparse
import struct

# Prepends operand to the name to avoid conflicts with other variables.
def operand_js_name(name):
   return "operand_" + js_name(name);

# Sanitize the name to be a valid JavaScript variable name
def js_name(name):
   return name.replace(".", "_").replace("::", "_").replace("/", "_");

used_variable_names = set()
# Adds the keyword let, if the name has not been used before.
def prepend_let(name):
   if name in used_variable_names:
      return name;
   else:
      used_variable_names.add(name)
      return "let " + name;

weights_file = None
last_bin_file_pos = 0;
def get_weights_and_biases_operand(name, model_proto):
  global last_bin_file_pos
  for ten_proto in model_proto.graph.initializer:
      if ten_proto.name == name:
          weights = onnx.numpy_helper.to_array(ten_proto)
          weights = weights.ravel()
          weights_bytes = weights.tobytes()
          weights_file.write(weights_bytes)
          binary_size = len(weights_bytes)
          print(f"{prepend_let('operand_value')} = new Float32Array(weights_buffer, {last_bin_file_pos}, {int(binary_size/4)});")
          last_bin_file_pos = last_bin_file_pos + binary_size
          operandDesc = f"{prepend_let('operand_desc')} = {{type: 'float32', dataType: 'float32', dimensions: {str(ten_proto.dims)}}};"
          print(operandDesc)
          declaration = f"const {operand_js_name(ten_proto.name)} = builder.constant(operand_desc, operand_value);"
          print(declaration)
  return operand_js_name(name)

def generateConv2D(node, model_proto):
   print(f"{prepend_let('conv2d_options')} = {{}};")
   layer_input_name = ""
   layer_filter_name = ""
   for input_name in node.input:
      if ".weight" in input_name:
         layer_filter_name = get_weights_and_biases_operand(
             input_name, model_proto)
      elif ".bias" in input_name:
         print(
             f"conv2d_options.bias = {get_weights_and_biases_operand(input_name, model_proto)}")
      else:
         layer_input_name = operand_js_name(input_name)

   for attribute in node.attribute:
      match attribute.name:
         case "group":
            print(f"conv2d_options.groups = {attribute.i};")
         case "pads":
            print(f"conv2d_options.padding = {attribute.ints};")
         case "strides":
            print(f"conv2d_options.strides = {attribute.ints};")
         case "dilations":
            print(f"conv2d_options.dilations = {attribute.ints};")
   print(
       f"let {operand_js_name(node.output[0])} = builder.conv2d({layer_input_name}, {layer_filter_name}, conv2d_options);")
   return

def generateUnary(op, node, model_proto):
   print(
       f"const {operand_js_name(node.output[0])} = builder.{op}({operand_js_name(node.input[0])})")

def generateBinary(op, node, model_proto):
   print(
       f"const {operand_js_name(node.output[0])} = builder.{op}({operand_js_name(node.input[0])}, {operand_js_name(node.input[1])})")

def generateGobalAveragePool(node, model_proto):
   print(
       f"{prepend_let('average_pool_window')} = {operand_js_name(node.input[0])}.shape().slice(-2);")
   print(
       f"{prepend_let('average_pool_output_size')} = {operand_js_name(node.input[0])}.shape().slice(0, -2);")
   print(f"{prepend_let('pool2d_options')} = {{}};")
   print("pool2d_options.windowDimensions = average_pool_window")
   print(
       f"const {operand_js_name(node.output[0])} = builder.averagePool2d({operand_js_name(node.input[0])}, pool2d_options)")

def generateDepthToSpace(node, model_proto):
   mode = ""
   blocksize = 0
   for attribute in node.attribute:
      match attribute.name:
         case "mode":
            mode = attribute.s.decode('UTF-8')
         case "blocksize":
            blocksize = attribute.i
   if mode != "CRD":
      print("// Only CRD is supported for Depth to space")
      sys.exit()

   print(
       f"{prepend_let('depth_to_space_shape')} = {operand_js_name(node.input[0])}.shape();")
   print(
       f"{prepend_let('depth_to_space_tmp')} = builder.reshape({operand_js_name(node.input[0])}, [depth_to_space_shape[0], depth_to_space_shape[1] / {blocksize * blocksize}, {blocksize}, {blocksize}, depth_to_space_shape[2], depth_to_space_shape[3]])")
   print(f"{prepend_let('transpose_options')} = {{}}")
   print("transpose_options.permutation = [0, 1, 4, 2, 5, 3]")
   print("depth_to_space_tmp = builder.transpose(depth_to_space_tmp, transpose_options)");
   print(f"{operand_js_name(node.output[0])} = builder.reshape(depth_to_space_tmp, [depth_to_space_shape[0], depth_to_space_shape[1] / {blocksize * blocksize}, depth_to_space_shape[2] * {blocksize}, depth_to_space_shape[3] * {blocksize}])")

def generateConstant(node, model_proto):
   global last_bin_file_pos
   if (len(node.attribute[0].t.dims) > 1):
      shape = f"[{', '.join(map(str, node.attribute[0].t.dims))}]"
      if node.attribute[0].t.data_type == 7:
         bytes_written = weights_file.write(node.attribute[0].t.raw_data)
         print(f"{operand_js_name(node.output[0])} = builder.constant({{dataType: 'int64', dimensions: {shape}}}, new BigInt64Array(weights_buffer, {last_bin_file_pos}, {int(bytes_written/8)}))");
         last_bin_file_pos = last_bin_file_pos + bytes_written
      elif node.attribute[0].t.data_type == 1:
         bytes_written = weights_file.write(node.attribute[0].t.raw_data)
         print(f"{operand_js_name(node.output[0])} = builder.constant({{dataType: 'int64', dimensions: {shape}}}, new Float32Array(weights_buffer, {last_bin_file_pos}, {int(bytes_written/4)}))");
         last_bin_file_pos = last_bin_file_pos + bytes_written
      else:
         # We dont support more than 1D array constant.
         terminateForUnsupportedNode(node);
   else:
      if node.attribute[0].t.data_type == 7:
         if (node.attribute[0].t.dims and node.attribute[0].t.dims[0] != 1 and len(node.attribute[0].t.dims) > 1):
            # We dont support constant integer arrays yet.
            terminateForUnsupportedNode(node);
         print(f"const {operand_js_name(node.output[0])} = {int.from_bytes(node.attribute[0].t.raw_data, byteorder='little')}");
      elif node.attribute[0].t.data_type == 1:
         dims = 1;
         if node.attribute[0].t.dims:
            dims = node.attribute[0].t.dims[0];
         format_string = f'<{dims}f'
         float_value = struct.unpack(format_string, node.attribute[0].t.raw_data)
         if (len(float_value) == 1):
            float_value = float_value[0]
         else:
            float_value = f"[{', '.join(map(str, float_value))}]"
         print(f"const {operand_js_name(node.output[0])} = {float_value}");
      else:
         terminateForUnsupportedNode(node);

def terminateForUnsupportedNode(node):
   print("// Unsupported Node {}!".format(node.op_type), file=sys.stderr)
   print("/*", file=sys.stderr)
   print(node, file=sys.stderr)
   print("*/", file=sys.stderr)
   print("}")
   sys.exit()

# Note: Shape is supported as outputting a CPU side operand only.
def generateShape(node, model_proto):
    print(f"{prepend_let(operand_js_name(node.output[0]))} = {operand_js_name(node.input[0])}.shape();");

def generateGather(node, model_proto):
    # default value of axis as per onnx is 0.
    axis = 0;
    if len(node.attribute) != 0:
      if node.attribute[0].name != "axis":
         terminateForUnsupportedNode(node);
      else:
         axis = node.attribute[0].i;
    print(f"{prepend_let(operand_js_name(node.output[0]))} = builder.gather({operand_js_name(node.input[0])}, {operand_js_name(node.input[1])}, {{axis:{axis}}})");

def generateCast(node, model_proto):
    if node.attribute[0].i == 7:
       dest_datatype = "int64";
    else:
       terminateForUnsupportedNode(node);
    print(f"{prepend_let(operand_js_name(node.output[0]))} = builder.cast({operand_js_name(node.input[0])}, \"{dest_datatype}\")");

def generateUnsqueeze(node, model_proto):
    # No where in the spec does it call out that default for axis is 0.
    # using that value for now.
    axis = 0;
    if len(node.attribute) != 0 :
      if node.attribute[0].name == "axes" and len(node.attribute[0].ints) == 1:
         axis = node.attribute[0].ints[0];
      else:
         terminateForUnsupportedNode(node);
    print(f"{prepend_let(operand_js_name(node.output[0]))} = builder.unsqueeze({operand_js_name(node.input[0])}, {axis})");

def generateConcat(node, model_proto):
    if node.attribute[0].name != "axis":
      terminateForUnsupportedNode(node);
    sequence = ", ".join(list(map(operand_js_name, node.input)));
    print(f"{prepend_let(operand_js_name(node.output[0]))} = builder.concat([{sequence}], {node.attribute[0].i})");

def generateTranspose(node, model_proto):
    if node.attribute[0].name != "perm":
      terminateForUnsupportedNode(node);
    permutation = str(node.attribute[0].ints);
    print(f"{prepend_let(operand_js_name(node.output[0]))} = builder.transpose({operand_js_name(node.input[0])}, {{ permutation: {permutation} }})");

def generateLeakyRelu(node, model_proto):
    if node.attribute[0].name != "alpha":
      terminateForUnsupportedNode(node);
    permutation = str(node.attribute[0].ints);
    print(f"{prepend_let(operand_js_name(node.output[0]))} = builder.leakyRelu({operand_js_name(node.input[0])}, {{ alpha: {node.attribute[0].f} }})");

def generateResize(node, model_proto):
   mode = None;
   for at in node.attribute:
      if at.name == "mode":
         mode = at.s;
         break;
   if mode != b'nearest':
      terminateForUnsupportedNode(node);
   print(f"{prepend_let(operand_js_name(node.output[0]))} = builder.resample2d({operand_js_name(node.input[0])}, {{ scales: [{operand_js_name(node.input[2])}[2], {operand_js_name(node.input[2])}[3]] }})");

def translate_node_to_webnn(node, model_proto):
   match node.op_type:
    case "Conv":
       generateConv2D(node, model_proto);
    case "Relu":
       generateUnary("relu", node, model_proto);
    case "GlobalAveragePool":
       generateGobalAveragePool(node, model_proto);
    case "Sigmoid":
       generateUnary("sigmoid", node, model_proto);
    case "Mul":
       generateBinary("mul", node, model_proto);
    case "Div":
       generateBinary("div", node, model_proto);
    case "Add":
       generateBinary("add", node, model_proto);
    case "DepthToSpace":
       generateDepthToSpace(node, model_proto);
    case "Constant":
       generateConstant(node, model_proto);
    case "Shape":
       generateShape(node, model_proto);
    case "Gather":
       generateGather(node, model_proto);
    case "Identity":
       generateUnary("identity", node, model_proto);
    case "Cast":
       generateCast(node, model_proto);
    case "Unsqueeze":
       generateUnsqueeze(node, model_proto);
    case "Concat":
       generateConcat(node, model_proto);
    case "Reshape":
       generateBinary("reshape", node, model_proto);
    case "Transpose":
       generateTranspose(node, model_proto);
    case "LeakyRelu":
       generateLeakyRelu(node, model_proto); 
    case "Resize":
       generateResize(node, model_proto);
    case _:
       terminateForUnsupportedNode(node);


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("model_file", help="Path to the ONNX model file")
   parser.add_argument("--weights_file", default="weights.bin",
                  help="Path to the output weights file")
   parser.add_argument("--output_file", default="model.js",
                  help="Path to the output JavaScript file")
   args = parser.parse_args()

   weights_file = open(args.weights_file, "wb")
   model_file = args.model_file
   model_proto = onnx.load(model_file)
   operators = set()

   output_file = open(args.output_file, "w")
   sys.stdout = output_file

   print("function loadModelGraph(operand_input, weights_buffer, builder) {")

   # Traverse each node in the graph
   last_node = None
   for node in model_proto.graph.node:
      operators.add(node.op_type)
      # Convert each node to WebNN code
      translate_node_to_webnn(node, model_proto)
      last_node = node

   if last_node is not None:
      print(f"return {operand_js_name(last_node.output[0])};")

   print("}")

   weights_file.close()
   output_file.close()