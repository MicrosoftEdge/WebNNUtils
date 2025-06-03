# OnnxConverter

OnnxConverter a WebNN compiler for producing pure JS models from .onnx models.

## Introduction
While JS ML frameworks can evaluate models in the browser, they involve shipping a framework and expensive load time preprocessing that impacts latency. During preprocessing frameworks determine the input shapes for operators, partition operators to those that need to run on CPU and optimize the model graph.

With OnnxConverter there is no such overhead. At compile time OnnxConverter takes a static .onnx file and emits JS code that will build an 
equivalent WebNN graph. The resulting JS code can be used in the browser. In other words OnnxConverter code gens a JS WebNN graph building
function with the following signature based on a .onnx file. The builder parameter is the MLGraphBuilder WebNN javascript object.

```
function loadModelGraph(operand_input, weights_buffer, builder) {...}
```

The weights from the .onnx model are emitted as a .bin file, which 
the calling code needs to download and pass in as an ArrayBuffer.


```
const weights_file = 'weights.bin';
let cache = await caches.open("weights")
let weights_response = await cache.match(weights_file);
if (!weights_response)
{
    await cache.add(weights_file);
    weights_response = await cache.match(weights_file);
}
const weights_buffer = await weights_response.arrayBuffer();
```

### How is shape inferencing handled?
There is no shape inferencing, however the WebNN graph builder is polyfilled with methods that take non tensor, number array inputs. 

For example, an onnx graph that has 
```
// Pseudo code
a = tensor.shape();
b = a * 2 
```
will simply generate webnn graph nodes that call shape and the mul operator. This implies that the mul operation would typically raise an exception, as it expects tensor inputs rather than JavaScript numbers, which are returned by shape operators. The polyfill addresses this issue by enabling the mul operation to handle JavaScript numbers as valid inputs.

**CpuOps.js** defines all the polyfill ops needed to augment **MLGraphBuilder** with operators that work on JS Number. Before 
loadModelGraph is called, the polyfill needs to be installed with

```
const context = await navigator.ml.createContext({'deviceType' : 'gpu'});
const builder = new MLGraphBuilder(context);
InstallCpuOps(builder);
```

### Other Challenges
Generating the JS graph has some challenges, and they are addressed by adding separate passes over generated code.

#### OnnxConverter.py 
Main compiler that emits JS graph building code based on the onnx file.

#### ReorderModel.py 
In some models, graph traversal results in code gen where the input operands are computed after the point they are needed in. ReorderModel.py fixes the generated code for such models by reordering lines of code to ensure that inputs are available before being used.

#### CPUGraphPartitioner.py
For some models the outputs of certain ops need to return CPU values. This is because those values may be used later in operators 
that are available only on the CPU. CPUGraphPartitioner.py walks through the generated code and annotates such ops that need to
produce CPU values with cpu_ prefix. These ops will then used the polyfilled version of the op from CpuOps.js.

Why do we have non cpu_ annotated software ops then ? These are either for decomposition or to handle this case where results of
a shape operation are processed in a graph. Results of a shape operation are retained on CPU as long as possible.

### Summary
The way to use OnnxConverter is to first run OnnxConverter.py > ReorderModel.py > CPUGraphPartitioner.py. 
The resulting file can be referenced in your Html and graph loaded with loadModelGraph.
