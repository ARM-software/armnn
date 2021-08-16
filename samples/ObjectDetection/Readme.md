# Object Detection Example

## Introduction
This is a sample code showing object detection using Arm NN public C++ API. The compiled application can take

 * a video file

as input and
 * save a video file
 * or output video stream to the window

with detections shown in bounding boxes, class labels and confidence.

## Dependencies

This example utilises OpenCV functions to capture and output video data. Top level inference API is provided by Arm NN
library.

### Arm NN

Object detection example build system does not trigger Arm NN compilation. Thus, before building the application,
please ensure that Arm NN libraries and header files are available on your build platform.
The application executable binary dynamically links with the following Arm NN libraries:
* libarmnn.so
* libarmnnTfLiteParser.so

The build script searches for available Arm NN libraries in the following order:
1. Inside custom user directory specified by ARMNN_LIB_DIR cmake option.
2. Inside the current Arm NN repository, assuming that Arm NN was built following [this instructions](../../BuildGuideCrossCompilation.md).
3. Inside default locations for system libraries, assuming Arm NN was installed from deb packages.

Arm NN header files will be searched in parent directory of found libraries files under `include` directory, i.e.
libraries found in `/usr/lib` or `/usr/lib64` and header files in `/usr/include` (or `${ARMNN_LIB_DIR}/include`).

Please see [find_armnn.cmake](./cmake/find_armnn.cmake) for implementation details.

### OpenCV

This application uses [OpenCV (Open Source Computer Vision Library)](https://opencv.org/) for video stream processing.
Your host platform may have OpenCV available through linux package manager. If this is the case, please install it using
standard way. If not, our build system has a script to download and cross-compile required OpenCV modules
as well as [FFMPEG](https://ffmpeg.org/) and [x264 encoder](https://www.videolan.org/developers/x264.html) libraries.
The latter will build limited OpenCV functionality and application will support only video file input and video file output
way of working. Displaying video frames in a window requires building OpenCV with GTK and OpenGL support.

The application executable binary dynamically links with the following OpenCV libraries:
* libopencv_core.so.4.0.0
* libopencv_imgproc.so.4.0.0
* libopencv_imgcodecs.so.4.0.0
* libopencv_videoio.so.4.0.0
* libopencv_video.so.4.0.0
* libopencv_highgui.so.4.0.0

and transitively depends on:
* libavcodec.so (FFMPEG)
* libavformat.so (FFMPEG)
* libavutil.so (FFMPEG)
* libswscale.so (FFMPEG)
* libx264.so (x264)

The application searches for above libraries in the following order:
1. Inside custom user directory specified by OPENCV_LIB_DIR cmake option.
2. Inside default locations for system libraries.

If no OpenCV libraries were found, the cross-compilation build is extended with x264, ffmpeg and OpenCV compilation steps.

Note: Native build does not add third party libraries to compilation.

Please see [find_opencv.cmake](./cmake/find_opencv.cmake) for implementation details.

## Building
There are two flows for building this application:
* native build on a host platform,
* cross-compilation for a Arm-based host platform.

### Build Options

* CMAKE_TOOLCHAIN_FILE - choose one of the  available cross-compilation toolchain files:
    * `cmake/aarch64-toolchain.cmake`
    * `cmake/arm-linux-gnueabihf-toolchain.cmake`
* ARMNN_LIB_DIR - point to the custom location of the Arm NN libs and headers.
* OPENCV_LIB_DIR  - point to the custom location of the OpenCV libs and headers.
* BUILD_UNIT_TESTS -  set to `1` to build tests. Additionally to the main application, `object_detection_example-tests`
unit tests executable will be created.

### Native Build
To build this application on a host platform, firstly ensure that required dependencies are installed:
For example, for raspberry PI:
```commandline
sudo apt-get update
sudo apt-get -yq install pkg-config
sudo apt-get -yq install libgtk2.0-dev zlib1g-dev libjpeg-dev libpng-dev libxvidcore-dev libx264-dev
sudo apt-get -yq install libavcodec-dev libavformat-dev libswscale-dev
```

To build demo application, create a build directory:
```commandline
mkdir build
cd build
```
If you have already installed Arm NN and OpenCV:

Inside build directory, run cmake and make commands:
```commandline
cmake  ..
make
```
This will build the following in bin directory:
* object_detection_example - application executable

If you have custom Arm NN and OpenCV location, use `OPENCV_LIB_DIR` and `ARMNN_LIB_DIR` options:
```commandline
cmake  -DARMNN_LIB_DIR=/path/to/armnn -DOPENCV_LIB_DIR=/path/to/opencv ..
make
```

### Cross-compilation

This section will explain how to cross-compile the application and dependencies on a Linux x86 machine
for arm host platforms.

You will require working cross-compilation toolchain supported by your host platform. For raspberry Pi 3 and 4 with glibc
runtime version 2.28, the following toolchains were successfully used:
* https://releases.linaro.org/components/toolchain/binaries/latest-7/aarch64-linux-gnu/
* https://releases.linaro.org/components/toolchain/binaries/latest-7/arm-linux-gnueabihf/

Choose aarch64-linux-gnu if `lscpu` command shows architecture as aarch64 or arm-linux-gnueabihf if detected
architecture is armv71.

You can check runtime version on your host platform by running:
```
ldd --version
```
On **build machine**, install C and C++ cross compiler toolchains and add them to the PATH variable.

Install package dependencies:
```commandline
sudo apt-get update
sudo apt-get -yq install pkg-config
```
Package config is required by OpenCV build to discover FFMPEG libs.

To build demo application, create a build directory:
```commandline
mkdir build
cd build
```
Inside build directory, run cmake and make commands:

**Arm 32bit**
```commandline
cmake -DARMNN_LIB_DIR=<path-to-armnn-libs> -DCMAKE_TOOLCHAIN_FILE=cmake/arm-linux-gnueabihf-toolchain.cmake ..
make
```
**Arm 64bit**
```commandline
cmake -DARMNN_LIB_DIR=<path-to-armnn-libs> -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-toolchain.cmake ..
make
```

Add `-j` flag to the make command to run compilation in multiple threads.

From the build directory, copy the following to the host platform:
* bin directory - contains object_detection_example executable,
* lib directory - contains cross-compiled OpenCV, ffmpeg, x264 libraries,
* Your Arm NN libs used during compilation.

The full list of libs after cross-compilation to copy on your board:
```
libarmnn.so
libarmnn.so.28
libarmnn.so.28.0
libarmnnTfLiteParser.so
libarmnnTfLiteParser.so.24.4
libavcodec.so
libavcodec.so.58
libavcodec.so.58.54.100
libavdevice.so
libavdevice.so.58
libavdevice.so.58.8.100
libavfilter.so
libavfilter.so.7
libavfilter.so.7.57.100
libavformat.so
libavformat.so.58
libavformat.so.58.29.100
libavutil.so
libavutil.so.56
libavutil.so.56.31.100
libopencv_core.so
libopencv_core.so.4.0
libopencv_core.so.4.0.0
libopencv_highgui.so
libopencv_highgui.so.4.0
libopencv_highgui.so.4.0.0
libopencv_imgcodecs.so
libopencv_imgcodecs.so.4.0
libopencv_imgcodecs.so.4.0.0
libopencv_imgproc.so
libopencv_imgproc.so.4.0
libopencv_imgproc.so.4.0.0
libopencv_video.so
libopencv_video.so.4.0
libopencv_video.so.4.0.0
libopencv_videoio.so
libopencv_videoio.so.4.0
libopencv_videoio.so.4.0.0
libpostproc.so
libpostproc.so.55
libpostproc.so.55.5.100
libswresample.a
libswresample.so
libswresample.so.3
libswresample.so.3.5.100
libswscale.so
libswscale.so.5
libswscale.so.5.5.100
libx264.so
libx264.so.160
```
## Executing

Once the application executable is built, it can be executed with the following options:
* --video-file-path: Path to the video file to run object detection on **[REQUIRED]**
* --model-file-path: Path to the Object Detection model to use **[REQUIRED]**
* --label-path: Path to the label set for the provided model file **[REQUIRED]**
* --model-name: The name of the model being used. Accepted options: SSD_MOBILE | YOLO_V3_TINY **[REQUIRED]**
* --output-video-file-path: Path to the output video file with detections added in. Defaults to /tmp/output.avi
 **[OPTIONAL]**
* --preferred-backends: Takes the preferred backends in preference order, separated by comma.
                        For example: CpuAcc,GpuAcc,CpuRef. Accepted options: [CpuAcc, CpuRef, GpuAcc].
                        Defaults to CpuRef **[OPTIONAL]**

### Object Detection on a supplied video file

To run object detection on a supplied video file and output result to a video file:
```commandline
LD_LIBRARY_PATH=/path/to/armnn/libs:/path/to/opencv/libs ./object_detection_example --label-path /path/to/labels/file
 --video-file-path /path/to/video/file --model-file-path /path/to/model/file
 --model-name [YOLO_V3_TINY | SSD_MOBILE] --output-video-file-path /path/to/output/file
```

To run object detection on a supplied video file and output result to a window gui:
```commandline
LD_LIBRARY_PATH=/path/to/armnn/libs:/path/to/opencv/libs ./object_detection_example --label-path /path/to/labels/file
 --video-file-path /path/to/video/file --model-file-path /path/to/model/file
 --model-name [YOLO_V3_TINY | SSD_MOBILE]
```

This application has been verified to work against the MobileNet SSD and the YOLO V3 tiny models, which can be downloaded along with their label sets from the Arm Model Zoo:
* https://github.com/ARM-software/ML-zoo/tree/master/models/object_detection/ssd_mobilenet_v1
* https://github.com/ARM-software/ML-zoo/tree/master/models/object_detection/yolo_v3_tiny

---

# Application Overview
This section provides a walkthrough of the application, explaining in detail the steps:
1. Initialisation
    1. Reading from Video Source
    2. Preparing Labels and Model Specific Functions
2. Creating a Network
    1. Creating Parser and Importing Graph
    3. Optimizing Graph for Compute Device
    4. Creating Input and Output Binding Information
3. Object detection pipeline
    1. Pre-processing the Captured Frame
    2. Making Input and Output Tensors
    3. Executing Inference
    4. Postprocessing
    5. Decoding and Processing Inference Output
    6. Drawing Bounding Boxes


### Initialisation

##### Reading from Video Source
After parsing user arguments, the chosen video file or stream is loaded into an OpenCV `cv::VideoCapture` object.
We use [`IFrameReader`](./include/IFrameReader.hpp) interface and OpenCV specific implementation
[`CvVideoFrameReader`](./include/CvVideoFrameReader.hpp) in our main function to capture frames from the source using the
`ReadFrame()` function.

The `CvVideoFrameReader` object also tells us information about the input video. Using this information and application
arguments, we create one of the implementations of the [`IFrameOutput`](./include/IFrameOutput.hpp) interface:
[`CvVideoFileWriter`](./include/CvVideoFileWriter.hpp) or [`CvWindowOutput`](./include/CvWindowOutput.hpp).
This object will be used at the end of every loop to write the processed frame to an output video file or gui
window.
`CvVideoFileWriter` uses `cv::VideoWriter` with ffmpeg backend. `CvWindowOutput` makes use of `cv::imshow()` function.

See `GetFrameSourceAndSink` function in [Main.cpp](./src/Main.cpp) for more details.

##### Preparing Labels and Model Specific Functions
In order to interpret the result of running inference on the loaded network, it is required to load the labels
associated with the model. In the provided example code, the `AssignColourToLabel` function creates a vector of pairs
label - colour that is ordered according to object class index at the output node of the model. Labels are assigned with
a randomly generated RGB color. This ensures that each class has a unique color which will prove helpful when plotting
the bounding boxes of various detected objects in a frame.

Depending on the model being used, `CreatePipeline`  function returns specific implementation of the object detection
pipeline.

### Creating a Network

All operations with Arm NN and networks are encapsulated in [`ArmnnNetworkExecutor`](./include/ArmnnNetworkExecutor.hpp)
class.

##### Creating Parser and Importing Graph
The first step with Arm NN SDK is to import a graph from file by using the appropriate parser.

The Arm NN SDK provides parsers for reading graphs from a variety of model formats. In our application we specifically
focus on `.tflite, .pb, .onnx` models.

Based on the extension of the provided model file, the corresponding parser is created and the network file loaded with
`CreateNetworkFromBinaryFile()` method. The parser will handle the creation of the underlying Arm NN graph.

Current example accepts tflite format model files, we use `ITfLiteParser`:
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

### Object detection pipeline

Generic object detection pipeline has 3 steps to perform data pre-processing, run inference and decode inference results
in the post-processing step.

See [`ObjDetectionPipeline`](include/ObjectDetectionPipeline.hpp) and implementations for [`MobileNetSSDv1`](include/ObjectDetectionPipeline.hpp)
and [`YoloV3Tiny`](include/ObjectDetectionPipeline.hpp) for more details.

#### Pre-processing the Captured Frame
Each frame captured from source is read as an `cv::Mat` in BGR format but channels are swapped to RGB in a frame reader
code.

```c++
cv::Mat processed;
...
objectDetectionPipeline->PreProcessing(frame, processed);
```

A pre-processing step consists of resizing the frame to the required resolution, padding  and doing data type conversion
to match the model input layer.
For example, SSD MobileNet V1 that is used in our example takes for input a tensor with shape `[1, 300, 300, 3]` and
data type `uint8`.

Pre-processing step returns `cv::Mat` object containing data ready for inference.

#### Executing Inference
```c++
od::InferenceResults results;
...
objectDetectionPipeline->Inference(processed, results);
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

##### Decoding and Processing Inference Output
The output from inference must be decoded to obtain information about detected objects in the frame. In the examples
there are implementations for two networks but you may also implement your own network decoding solution here.

For SSD MobileNet V1 models, we decode the results to obtain the bounding box positions, classification index,
confidence and number of detections in the input frame.
See [`SSDResultDecoder`](./include/SSDResultDecoder.hpp) for more details.

For YOLO V3 Tiny models, we decode the output and perform non-maximum suppression to filter out any weak detections
below a confidence threshold and any redudant bounding boxes above an intersection-over-union threshold.
See [`YoloResultDecoder`](./include/YoloResultDecoder.hpp) for more details.

It is encouraged to experiment with threshold values for confidence and intersection-over-union (IoU)
to achieve the best visual results.

The detection results are always returned as a vector of [`DetectedObject`](./include/DetectedObject.hpp),
with the box positions list containing bounding box coordinates in the form `[x_min, y_min, x_max, y_max]`.

#### Drawing Bounding Boxes
Post-processing step accepts a callback function to be invoked when the decoding is finished. We will use it
to draw detections on the initial frame.
With the obtained detections and using [`AddInferenceOutputToFrame`](./src/ImageUtils.cpp) function, we are able to draw bounding boxes around
detected objects and add the associated label and confidence score.
```c++
//results - inference output
objectDetectionPipeline->PostProcessing(results, [&frame, &labels](od::DetectedObjects detects) -> void {
            AddInferenceOutputToFrame(detects, *frame, labels);
        });
```
The processed frames are written to a file or displayed in a separate window.