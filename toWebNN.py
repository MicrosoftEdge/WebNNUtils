import onnx
from onnx import helper
from onnx import TensorProto
import onnx.numpy_helper
import numpy as np
import sys

def operand_js_name(name):
   return "operand_" + js_name(name);
def js_name(name):
   return name.replace(".", "_").replace("::", "_");

var_name = set();
def prepend_let(name):
   if name in var_name:
      return name;
   else:
      var_name.add(name);
      return "let " + name;

def get_weights_and_biases_operand(name, model_proto):
  # Loop through graph and find the corresponding weights and biases
  for ten_proto in model_proto.graph.initializer:
      if ten_proto.name == name:
          weights = onnx.numpy_helper.to_array(ten_proto);
          print(prepend_let("operand_value") + " = new Float32Array(" + np.array2string(weights.ravel(), threshold=sys.maxsize, separator=",") + ");");
          #print(prepend_let("operand_value") + " = null;");
          operandDesc = prepend_let("operand_desc") + " = {type: 'float32', dataType: 'float32', dimensions: " + str(ten_proto.dims) + "};";
          print(operandDesc);
          declaration = "const " + operand_js_name(ten_proto.name) + " = builder.constant(operand_desc, operand_value);"
          print(declaration);
  return operand_js_name(name);

def generateConv2D(node, model_proto):
   print(prepend_let("conv2d_options") + " = {};" );
   layer_input_name = "";
   layer_filter_name = "";
   for input_name in node.input:
      if ".weight" in input_name:
         layer_filter_name = get_weights_and_biases_operand(input_name, model_proto);
      elif ".bias" in input_name:
         print("conv2d_options.bias = " + get_weights_and_biases_operand(input_name, model_proto));
      else:
         layer_input_name = operand_js_name(input_name);

   for attribute in node.attribute:
      match attribute.name:
         case "group":
            print("conv2d_options.groups = " + str(attribute.i) + ";");
         case "pads":
            print("conv2d_options.padding = " + str(attribute.ints) + ";");
         case "strides":
            print("conv2d_options.strides = " + str(attribute.ints) + ";");
         case "dilations":
            print("conv2d_options.dilations = " + str(attribute.ints) + ";");
   print("let " + operand_js_name(node.output[0]) + " = " + 
         "builder.conv2d(" + layer_input_name + ", " + layer_filter_name + ", conv2d_options);");
   return;


def generateUnary(op, node, model_proto):
   print("const " + operand_js_name(node.output[0]) + " = " + " builder."+ op + "(" + operand_js_name(node.input[0]) + ")");

def generateBinary(op, node, model_proto):
   print("const " + operand_js_name(node.output[0]) + " = " + " builder."+ op + "(" + operand_js_name(node.input[0]) + ", " + operand_js_name(node.input[1])+ ")");


def generateGobalAveragePool(node, model_proto):
   # We are assuming the spatial dimension are the last 2 in shape - works for nchw.
   print(f"{prepend_let('average_pool_window')} = {operand_js_name(node.input[0])}.shape().slice(-2);");
   print(f"{prepend_let('average_pool_output_size')} = {operand_js_name(node.input[0])}.shape().slice(0, -2);");
   print(f"{prepend_let('pool2d_options')} = {{}};");
   print(f"pool2d_options.windowDimensions = average_pool_window");
   #print(f"pool2d_options.outputSizes = average_pool_output_size");
   print("const " + operand_js_name(node.output[0]) + " = " + " builder.averagePool2d(" + operand_js_name(node.input[0]) + ", pool2d_options)");

def generateDepthToSpace(node, model_proto):
   mode = "";
   blocksize = 0;
   for attribute in node.attribute:
      match attribute.name:
         case "mode":
            mode = attribute.s.decode('UTF-8');
         case "blocksize":
            blocksize = attribute.i;
   if mode != "CRD":
      print("// Only CRD is supported for Depth to space");
      sys.exit();
   
   # For CRD
   # https://onnx.ai/onnx/operators/onnx__DepthToSpace.html
   # b, c, h, w = x.shape
   # tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])
   # tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
   # y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])
   print(f"{prepend_let('depth_to_space_shape')} = {operand_js_name(node.input[0])}.shape();");
   print(f"{prepend_let('depth_to_space_tmp')} = builder.reshape({operand_js_name(node.input[0])}, [depth_to_space_shape[0], depth_to_space_shape[1] / {blocksize * blocksize}, {blocksize}, {blocksize}, depth_to_space_shape[2], depth_to_space_shape[3]])")
   print(f"{prepend_let('transpose_options')} = {{}}") ;
   print("transpose_options.permutation = [0, 1, 4, 2, 5, 3]");
   print("depth_to_space_tmp = builder.transpose(depth_to_space_tmp, transpose_options)");
   print(f"{operand_js_name(node.output[0])} = builder.reshape(depth_to_space_tmp, [depth_to_space_shape[0], depth_to_space_shape[1] / {blocksize * blocksize}, depth_to_space_shape[2] * {blocksize}, depth_to_space_shape[3] * {blocksize}])")
   

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
    case "Add":
         generateBinary("add", node, model_proto);
    case "DepthToSpace":
         generateDepthToSpace(node, model_proto);
    case _:
         print("// Unsupported Node {}!".format(node.op_type))
         print("/*");
         print(node);
         print("*/");
         print("}")
         sys.exit();

# Load the ONNX model from file
model_file = "D:\\Projects\\WebML\\Onnxwebtest\\VideoSuperResolution-FP32.onnx";
model_proto = onnx.load(model_file)
operators = set();

print("function loadModelGraph(operand_input, builder) {")

# Traverse each node in the graph
last_node = None;
for node in model_proto.graph.node:
   operators.add(node.op_type);
   # Convert each node to WebNN code
   translate_node_to_webnn(node, model_proto)
   last_node = node;

if last_node != None:
   print(f"return {operand_js_name(last_node.output[0])};");

print("}")