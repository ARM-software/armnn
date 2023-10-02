# Interface tests for the release Arm NN binary package.

These are a small number of executables that exercise the main interfaces exposed by Arm NN via the binary packages. The intent is to highlight any missing include dependencies or libraries.

## Usage
The CMakeLists.txt file describes 5 binaries. Each focusing on a different interface. Before attempting to compile you must already have compiled the appropriate versions of Flat Buffers and Tensorflow Lite.

Standard practice for cmake is to create a subdirectory called 'build' and execute from within there.

```bash
mkdir build
cd build
cmake .. -DARMNN_ROOT=<path to the unpacked binary build> -DTFLITE_INCLUDE_ROOT=<directory containing tensorflow/include> -DTFLITE_LIB_ROOT=<directory containing libtensorflow-lite.a> -DFLATBUFFERS_ROOT=<directory containing flatbuffers install>
make
```

It is not strictly necessary to execute the built components as this is testing the interface to build rather than correctness of execution.

## Individual tests

### SimpleSample
This exercies the Arm NN graph interface. It is based on SimpleSample located in armnn/samples/SimpleSample.cpp. 

### TfLiteParserTest
This exercies the Arm NN TfLite parser interface. It will attempt to parse simple_conv2d_1_op.tflite, load it and execute an inference.

### OnnxParserTest
This exercies the Arm NN Onnx interface. It will attempt to parse a simple convoultion model hard coded in prototext, load it and execute an inference.

### ClassicDelegateTest / OpaqueDelegateTest
Neither of these tests are strictly necessary as the external interface is only used by the TfLite runtime. Users of Arm NN are never expected to hard code an execution of either TfLite delegate. Instead, the delegate library is presented to the TfLite runtime for it to execute. However, these tests exist just to verify this interface.
