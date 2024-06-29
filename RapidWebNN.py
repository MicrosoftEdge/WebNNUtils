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
   return name.replace(".", "_").replace(":", "_").replace("/", "_");

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

def handleWeightsAlignment(alignment):
   global last_bin_file_pos
   reminder = last_bin_file_pos%alignment;
   if (reminder !=0):
         # JS float32 arrays can be loaded only from aligned locations.
         # Fill in 0s to achieve alignment.
         weights_file.write(bytearray(alignment-reminder))
         last_bin_file_pos+=(alignment-reminder);

def generateInitializers(model_proto):
  global last_bin_file_pos
  for ten_proto in model_proto.graph.initializer:
      weights = onnx.numpy_helper.to_array(ten_proto)
      weights = weights.ravel()
      weights_bytes = weights.tobytes()
      binary_size = len(weights_bytes)

      dest_datatype = "";
      array_type = "";
      size = 0;
      alignment = 1;
      if ten_proto.data_type == 7:
         dest_datatype = "int64";
         array_type = "BigInt64Array"
         size = int(binary_size/8);
         alignment = 4;
      elif ten_proto.data_type == 1:
         dest_datatype = "float32";
         array_type = "Float32Array"
         size = int(binary_size/4)
         alignment = 4;
      elif ten_proto.data_type == 9:
         # 9 is supposed to be bool but there is no way to represent that in webnn
         dest_datatype = "uint8";
         array_type = "Uint8Array"
         size = int(binary_size)
      elif ten_proto.data_type == 2:
         dest_datatype = "uint8";
         array_type = "Uint8Array"
         size = int(binary_size)
      elif ten_proto.data_type == 3:
         dest_datatype = "int8";
         array_type = "Int8Array"
         size = int(binary_size)
      else:
         print("Unsupported Initializer !!",file=sys.stderr)
         print(ten_proto, file=sys.stderr);
         sys.exit();
      handleWeightsAlignment(alignment);
      # Write to the weights file.
      weights_file.write(weights_bytes)

      print(f"{prepend_let('operand_value')} = new {array_type}(weights_buffer, {last_bin_file_pos}, {size});")
      last_bin_file_pos = last_bin_file_pos + binary_size
      operandDesc = f"{prepend_let('operand_desc')} = {{type: '{dest_datatype}', dataType: '{dest_datatype}', dimensions: {str(ten_proto.dims)}}};"
      print(operandDesc)
      declaration = f"const {operand_js_name(ten_proto.name)} = builder.constant(operand_desc, operand_value);"
      print(declaration)


def generateConv2D(node, model_proto):
   print(f"{prepend_let('conv2d_options')} = {{}};")
   layer_input_name = ""
   layer_filter_name = ""
   for input_name in node.input:
      if ".weight" in input_name:
         layer_filter_name = operand_js_name(input_name);
      elif ".bias" in input_name:
         print(f"conv2d_options.bias = {operand_js_name(input_name)}")
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

def generateWhere(node, model_proto):
   print(
       f"const {operand_js_name(node.output[0])} = builder.where({operand_js_name(node.input[0])}, {operand_js_name(node.input[1])}, {operand_js_name(node.input[2])})")

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
         handleWeightsAlignment(4);
         bytes_written = weights_file.write(node.attribute[0].t.raw_data)
         print(f"{operand_js_name(node.output[0])} = builder.constant({{dataType: 'int64', dimensions: {shape}}}, new BigInt64Array(weights_buffer, {last_bin_file_pos}, {int(bytes_written/8)}))");
         last_bin_file_pos = last_bin_file_pos + bytes_written
      elif node.attribute[0].t.data_type == 1:
         handleWeightsAlignment(4);
         bytes_written = weights_file.write(node.attribute[0].t.raw_data)
         print(f"{operand_js_name(node.output[0])} = builder.constant({{dataType: 'float32', dimensions: {shape}}}, new Float32Array(weights_buffer, {last_bin_file_pos}, {int(bytes_written/4)}))");
         last_bin_file_pos = last_bin_file_pos + bytes_written
      else:
         # We dont support more than 1D array constant.
         terminateForUnsupportedNode(node);
   else:
      if node.attribute[0].t.data_type == 7:
         if (node.attribute[0].t.dims and node.attribute[0].t.dims[0] != 1 and len(node.attribute[0].t.dims) > 1):
            # We dont support constant integer arrays yet.
            terminateForUnsupportedNode(node);
         print(f"let {operand_js_name(node.output[0])} = {int.from_bytes(node.attribute[0].t.raw_data, byteorder='little', signed=True)}");
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
         print(f"let {operand_js_name(node.output[0])} = {float_value}");
      else:
         terminateForUnsupportedNode(node);

def generateConstantOfShape(node, model_proto):
   type = '';
   if node.attribute[0].t.data_type == 7:
      type = 'int64';
   elif node.attribute[0].t.data_type == 1:
      type = 'float32'
   else :
      terminateForUnsupportedNode(node);
   generateConstant(node, model_proto);
   print(f"{operand_js_name(node.output[0])} = builder.generateConstantOfShape('{type}', {operand_js_name(node.output[0])}, {operand_js_name(node.input[0])})");

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
    elif node.attribute[0].i == 1:
       dest_datatype = "float32";
    elif node.attribute[0].i == 9:
       # 9 is supposed to be bool but there is no way to represent that in webnn
       dest_datatype = "uint8";
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
    else:
      if len(node.input) > 1:
         # When there are two inputs the convention seems to be to treat the
         # second one as axis.
         axis = operand_js_name(node.input[1])
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

def generateDequantizeLinear(node, model_proto):
   if "_scale" not in node.input[1] or "_zero_point" not in node.input[2]:
      terminateForUnsupportedNode(node);
   # Proceeding as though the 2nd input is scale and 3rd input is zero point.
   scale_name = operand_js_name(node.input[1]);
   zero_point_name = operand_js_name(node.input[2]);
   #y = (x - x_zero_point) * x_scale
   print(f"{prepend_let(operand_js_name(node.output[0]))} = builder.sub({operand_js_name(node.input[0])}, {zero_point_name})");
   print(f"{operand_js_name(node.output[0])} = builder.mul(builder.cast({operand_js_name(node.output[0])}, 'float32'), {scale_name})");

def generateReduceMean(node, model_proto):
   keepDimensions = False;
   axes_string = "[";
   has_negative_axes = False;
   for at in node.attribute:
      if at.name == "keepdims":
         keepDimensions = (at.i == 1);
      elif at.name == "axes":
         for axis in at.ints:
            if axis < 0:
               axes_string += ("rank" + str(axis) + ",");
               has_negative_axes = True;
            else:
               axes_string += (axis + ",");
   axes_string += "]";
   if has_negative_axes:
      print(f"{prepend_let('rank')} = {operand_js_name(node.input[0])}.shape().length;");
   if keepDimensions:
      keepDimensionsString = "true";
   else:
      keepDimensionsString = "false";
   print(f"{prepend_let(operand_js_name(node.output[0]))} = builder.reduceMean({operand_js_name(node.input[0])}, {{ keepDimensions: {keepDimensionsString}, axes: {axes_string} }})");

# Not a real implementation of slice, just passing through values
# so that slice can be implemented in JS for the limited use case
# of slicing constants.
def generateSWSlice(node, model_proto):
   if len(node.input) == 5:
      print(f"{prepend_let(operand_js_name(node.output[0]))} = builder.slice({operand_js_name(node.input[0])}, {operand_js_name(node.input[1])}, {operand_js_name(node.input[2])}, {operand_js_name(node.input[3])}, {operand_js_name(node.input[4])})");
   elif len(node.input) == 4:
      print(f"{prepend_let(operand_js_name(node.output[0]))} = builder.slice({operand_js_name(node.input[0])}, {operand_js_name(node.input[1])}, {operand_js_name(node.input[2])}, {operand_js_name(node.input[3])})");
   elif len(node.input) == 3:
      print(f"{prepend_let(operand_js_name(node.output[0]))} = builder.slice({operand_js_name(node.input[0])}, {operand_js_name(node.input[1])}, {operand_js_name(node.input[2])})");
   else:
      terminateForUnsupportedNode(node);

# Limited implementation of squeeze that is actually implemented on the JS side.
def generateSWSqueeze(node, model_proto):
   print(f"{prepend_let(operand_js_name(node.output[0]))} = builder.squeeze({operand_js_name(node.input[0])}, {operand_js_name(node.input[1])})");

# Limited implementation of range that is actually implemented on the JS side.
def generateSWRange(node, model_proto):
   print(f"{prepend_let(operand_js_name(node.output[0]))} = builder.range({operand_js_name(node.input[0])}, {operand_js_name(node.input[1])}, {operand_js_name(node.input[2])})");

def generateExpand(node, model_proto):
   print(f"{prepend_let(operand_js_name(node.output[0]))} = builder.expand({operand_js_name(node.input[0])}, {operand_js_name(node.input[1])})");

def generateDynamicQuantizeLinear(node, model_proto):
   print(f"{prepend_let('max_dql')} = builder.reduceMax({operand_js_name(node.input[0])})");
   print(f"{prepend_let('min_dql')} = builder.reduceMin({operand_js_name(node.input[0])})");
   # y_scale = (max(x) - min(x))/(qmax - qmin)
   print(f"{prepend_let(operand_js_name(node.output[1]))} = builder.div(builder.sub(max_dql, min_dql), builder.constant_dql_255)");
   # intermediate_zero_point = qmin - min(x)/y_scale
   # y_zero_point = cast(round(saturate(itermediate_zero_point)))
   print(f"{prepend_let('izp_dql')} = builder.div(builder.sub(builder.constant_dql_255, min_dql), {operand_js_name(node.output[1])})");
   print(f"izp_dql = builder.clamp(izp_dql, {{minValue:0, maxValue:255}})");
   # Implement round by adding 0.5 before the cast.
   print(f"izp_dql = builder.add(izp_dql, builder.constant_dql_pt5)");
   print(f"{prepend_let(operand_js_name(node.output[2]))} = builder.cast(izp_dql, \"uint8\")");
   print(f"{prepend_let(operand_js_name(node.output[0]))} = builder.QuantizeLinear({operand_js_name(node.input[0])}, {operand_js_name(node.output[1])}, izp_dql);");

def generateMatMulInteger(node, model_proto):
   print(f"{prepend_let('intermediate_' + operand_js_name(node.input[0]))} = builder.sub(builder.cast({operand_js_name(node.input[0])}, 'int32'), builder.cast({operand_js_name(node.input[2])}, 'int32'))");
   print(f"{prepend_let('intermediate_' + operand_js_name(node.input[2]))} = builder.sub(builder.cast({operand_js_name(node.input[1])}, 'int32'), builder.cast({operand_js_name(node.input[3])}, 'int32'))");
   print(f"{prepend_let(operand_js_name(node.output[0]))} = builder.matmul({'intermediate_'+operand_js_name(node.input[0])}, {'intermediate_'+operand_js_name(node.input[2])})");
   
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
    case "Neg":
       generateUnary("neg", node, model_proto);
    case "Softmax":
       generateUnary("softmax", node, model_proto);
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
    case "DequantizeLinear":
       generateDequantizeLinear(node, model_proto);
    case "Pow":
       generateBinary("pow", node, model_proto);
    case "Sub":
       generateBinary("sub", node, model_proto);
    case "ConstantOfShape":
       generateConstantOfShape(node, model_proto);
    case "ReduceMean":
       generateReduceMean(node, model_proto);
    case "Slice":
       generateSWSlice(node, model_proto);
    case "Squeeze":
       generateSWSqueeze(node, model_proto);
    case "Sqrt":
       generateUnary("sqrt", node, model_proto);
    case "Equal":
       generateBinary("equal", node, model_proto);
    case "Range":
       generateSWRange(node, model_proto);
    case "Where":
       generateWhere(node, model_proto);
    case "Expand":
       generateExpand(node, model_proto);
    case "DynamicQuantizeLinear":
       generateDynamicQuantizeLinear(node, model_proto);
    case "Less":
       generateBinary("lesser", node, model_proto);
    case "MatMulInteger":
       generateMatMulInteger(node, model_proto);
    case _:
       terminateForUnsupportedNode(node);

def generateFunctionSignature(model_proto):
   print("function loadModelGraph(",  end ="");
   for inp in model_proto.graph.input:
      print(operand_js_name(inp.name) + ",");
   print("weights_buffer, builder) {");

def generateFunctionReturn(model_proto):
   print("return ",  end ="");
   if len(model_proto.graph.output) == 1:
      print(operand_js_name(model_proto.graph.output[0]));
   else:
      print("{",  end ="");
      for out in model_proto.graph.output:
         print("\""+out.name+"\":" + operand_js_name(out.name) + ",");
      print("}",  end ="");
   print(";");

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

   generateFunctionSignature(model_proto);
   generateInitializers(model_proto);

   # Traverse each node in the graph
   last_node = None
   for node in model_proto.graph.node:
      operators.add(node.op_type)
      # Convert each node to WebNN code
      translate_node_to_webnn(node, model_proto)
      last_node = node

   generateFunctionReturn(model_proto);

   print("}")

   weights_file.close()
   output_file.close()