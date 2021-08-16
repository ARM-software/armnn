# Keyword Spotting Example

## Introduction

This is a sample code showing keyword spotting using Arm NN public C++ API. The compiled application can take

* an audio file

as input and produce

* recognised keyword in the audio file

as output. The application works with the [fully quantised DS CNN Large model](https://github.com/ARM-software/ML-zoo/raw/68b5fbc77ed28e67b2efc915997ea4477c1d9d5b/models/keyword_spotting/ds_cnn_large/tflite_clustered_int8/) which is trained to recongize 12 keywords, including an unknown word.

## Dependencies

This example utilises `libsndfile`, `libasound` and `libsamplerate` libraries to capture the raw audio data from file, and to re-sample to the expected sample rate. Top level inference API is provided by Arm NN library.

### Arm NN

Keyword spotting example build system does not trigger Arm NN compilation. Thus, before building the application,
please ensure that Arm NN libraries and header files are available on your build platform.
The application executable binary dynamically links with the following Arm NN libraries:

* libarmnn.so
* libarmnnTfLiteParser.so

The build script searches for available Arm NN libraries in the following order:

1. Inside custom user directory specified by ARMNN_LIB_DIR cmake option.
2. Inside the current Arm NN repository, assuming that Arm NN was built following [these instructions](../../BuildGuideCrossCompilation.md).
3. Inside default locations for system libraries, assuming Arm NN was installed from deb packages.

Arm NN header files will be searched in parent directory of found libraries files under `include` directory, i.e.
libraries found in `/usr/lib` or `/usr/lib64` and header files in `/usr/include` (or `${ARMNN_LIB_DIR}/include`).

Please see [find_armnn.cmake](./cmake/find_armnn.cmake) for implementation details.

## Building

There is one flow for building this application:

* native build on a host platform

### Build Options

* ARMNN_LIB_DIR - point to the custom location of the Arm NN libs and headers.
* BUILD_UNIT_TESTS -  set to `1` to build tests. Additionally to the main application, `keyword-spotting-example-tests`
unit tests executable will be created.

### Native Build

To build this application on a host platform, firstly ensure that required dependencies are installed:
For example, for raspberry PI:

```commandline
sudo apt-get update
sudo apt-get -yq install libsndfile1-dev
sudo apt-get -yq install libasound2-dev
sudo apt-get -yq install libsamplerate-dev
```

To build demo application, create a build directory:

```commandline
mkdir build
cd build
```

If you have already installed Arm NN and and the required libraries:

Inside build directory, run cmake and make commands:

```commandline
cmake  ..
make
```

This will build the following in bin directory:

* `keyword-spotting-example` - application executable

If you have custom Arm NN location, use `ARMNN_LIB_DIR` options:

```commandline
cmake  -DARMNN_LIB_DIR=/path/to/armnn ..
make
```

## Executing

Once the application executable is built, it can be executed with the following options:

* --audio-file-path: Path to the audio file to run keyword spotting on **[REQUIRED]**
* --model-file-path: Path to the Keyword Spotting model to use **[REQUIRED]**

* --preferred-backends: Takes the preferred backends in preference order, separated by comma.
                        For example: `CpuAcc,GpuAcc,CpuRef`. Accepted options: [`CpuAcc`, `CpuRef`, `GpuAcc`].
                        Defaults to `CpuRef` **[OPTIONAL]**

### Keyword Spotting on a supplied audio file

A small selection of suitable wav files containing keywords can be found [here](https://git.mlplatform.org/ml/ethos-u/ml-embedded-evaluation-kit.git/plain/resources/kws/samples/).
To run keyword spotting on a supplied audio file and output the result to console:

```commandline
./keyword-spotting-example --audio-file-path /path/to/audio/file --model-file-path /path/to/model/file
```

# Application Overview

This section provides a walkthrough of the application, explaining in detail the steps:

1. Initialisation
    1. Reading from Audio Source
2. Creating a Network
    1. Creating Parser and Importing Graph
    2. Optimizing Graph for Compute Device
    3. Creating Input and Output Binding Information
3. Keyword spotting pipeline
    1. Pre-processing the Captured Audio
    2. Making Input and Output Tensors
    3. Executing Inference
    4. Postprocessing
    5. Decoding and Processing Inference Output

### Initialisation

##### Reading from Audio Source

After parsing user arguments, the chosen audio file is loaded into an AudioCapture object.
We use [`AudioCapture`](./include/AudioCapture.hpp) in our main function to capture appropriately sized audio blocks from the source using the
`Next()` function.

The `AudioCapture` object also re-samples the audio input to a desired sample rate, and sets the number of channels used to one channel (i.e `mono`)

### Creating a Network

All operations with Arm NN and networks are encapsulated in [`ArmnnNetworkExecutor`](./include/ArmnnNetworkExecutor.hpp)
class.

##### Creating Parser and Importing Graph

The first step with Arm NN SDK is to import a graph from file by using the appropriate parser.

The Arm NN SDK provides parsers for reading graphs from a variety of model formats. In our application we specifically
focus on `.tflite, .pb, .onnx` models.

Based on the extension of the provided model file, the corresponding parser is created and the network file loaded with
`CreateNetworkFromBinaryFile()` method. The parser will handle the creation of the underlying Arm NN graph.

Currently this example only supports tflite format model files and uses `ITfLiteParser`:

```c++
#include "armnnTfLiteParser/ITfLiteParser.hpp"

armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();
armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(modelPath.c_str());
```

##### Optimizing Graph for Compute Device

Arm NN supports optimized execution on multiple CPU and GPU devices. Prior to executing a graph, we must select the
appropriate device context. We do this by creating a runtime context with default options with `IRuntime()`.

For example:

```c++
#include "armnn/ArmNN.hpp"

auto runtime = armnn::IRuntime::Create(armnn::IRuntime::CreationOptions());
```

We can optimize the imported graph by specifying a list of backends in order of preference and implement
backend-specific optimizations. The backends are identified by a string unique to the backend,
for example `CpuAcc, GpuAcc, CpuRef`.

For example:

```c++
std::vector<armnn::BackendId> backends{"CpuAcc", "GpuAcc", "CpuRef"};
```

Internally and transparently, Arm NN splits the graph into subgraph based on backends, it calls a optimize subgraphs
function on each of them and, if possible, substitutes the corresponding subgraph in the original graph with
its optimized version.

Using the `Optimize()` function we optimize the graph for inference and load the optimized network onto the compute
device with `LoadNetwork()`. This function creates the backend-specific workloads
for the layers and a backend specific workload factory which is called to create the workloads.

For example:

```c++
armnn::IOptimizedNetworkPtr optNet = Optimize(*network,
                                              backends,
                                              m_Runtime->GetDeviceSpec(),
                                              armnn::OptimizerOptions());
std::string errorMessage;
runtime->LoadNetwork(0, std::move(optNet), errorMessage));
std::cerr << errorMessage << std::endl;
```

##### Creating Input and Output Binding Information

Parsers can also be used to extract the input information for the network. By calling `GetSubgraphInputTensorNames`
we extract all the input names and, with `GetNetworkInputBindingInfo`, bind the input points of the graph.
For example:

```c++
std::vector<std::string> inputNames = parser->GetSubgraphInputTensorNames(0);
auto inputBindingInfo = parser->GetNetworkInputBindingInfo(0, inputNames[0]);
```

The input binding information contains all the essential information about the input. It is a tuple consisting of
integer identifiers for bindable layers (inputs, outputs) and the tensor info (data type, quantization information,
number of dimensions, total number of elements).

Similarly, we can get the output binding information for an output layer by using the parser to retrieve output
tensor names and calling `GetNetworkOutputBindingInfo()`.

### Keyword Spotting pipeline

The keyword spotting pipeline has 3 steps to perform: data pre-processing, run inference and decode inference results.

See [`KeywordSpottingPipeline`](include/KeywordSpottingPipeline.hpp) for more details.

#### Pre-processing the Audio Input

Each frame captured from source is read and stored by the AudioCapture object.
It's `Next()` function provides us with the correctly positioned window of data, sized appropriately for the given model, to pre-process before inference.

```c++
std::vector<float> audioBlock = capture.Next();
...
std::vector<int8_t> preprocessedData = kwsPipeline->PreProcessing(audioBlock);
```

The `MFCC` class is then used to extract the Mel-frequency Cepstral Coefficients (MFCCs, [see Wikipedia](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)) from each stored audio frame in the provided window of audio, to be used as features for the network. MFCCs are the result of computing the dot product of the Discrete Cosine Transform (DCT) Matrix and the log of the Mel energy.

After all the MFCCs needed for an inference have been extracted from the audio data they are concatenated to make the input tensor for the model.

#### Executing Inference

```c++
common::InferenceResults results;
...
kwsPipeline->Inference(preprocessedData, results);
```

Inference step will call `ArmnnNetworkExecutor::Run` method that will prepare input tensors and execute inference.
A compute device performs inference for the loaded network using the `EnqueueWorkload()` function of the runtime context.
For example:

```c++
//const void* inputData = ...;
//outputTensors were pre-allocated before

armnn::InputTensors inputTensors = {{ inputBindingInfo.first,armnn::ConstTensor(inputBindingInfo.second, inputData)}};
runtime->EnqueueWorkload(0, inputTensors, outputTensors);
```

We allocate memory for output data once and map it to output tensor objects. After successful inference, we read data
from the pre-allocated output data buffer. See [`ArmnnNetworkExecutor::ArmnnNetworkExecutor`](./src/ArmnnNetworkExecutor.cpp)
and [`ArmnnNetworkExecutor::Run`](./src/ArmnnNetworkExecutor.cpp) for more details.

#### Postprocessing

##### Decoding

The output from the inference is decoded to obtain the spotted keyword- the word with highest probability is outputted to the console.

```c++
kwsPipeline->PostProcessing(results, labels,
                            [](int index, std::string& label, float prob) -> void {
                                printf("Keyword \"%s\", index %d:, probability %f\n",
                                        label.c_str(),
                                        index,
                                        prob);
                            });
```

The produced string is displayed on the console.
