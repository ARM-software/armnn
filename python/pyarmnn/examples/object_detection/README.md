# Object Detection Sample Application

## Introduction
This sample application guides the user and shows how to perform object detection using PyArmNN or Arm NN TensorFlow Lite Delegate API. We assume the user has already built PyArmNN by following the instructions of the README in the main PyArmNN directory.

##### Running with Armn NN TensorFlow Lite Delegate
There is an option to use the Arm NN TensorFlow Lite Delegate instead of Arm NN TensorFlow Lite Parser for the object detection inference.
The Arm NN TensorFlow Lite Delegate is part of Arm NN library and its purpose is to accelerate certain TensorFlow Lite
(TfLite) operators on Arm hardware. The main advantage of using the Arm NN TensorFlow Lite Delegate over the Arm NN TensorFlow
Lite Parser is that the number of supported operations is far greater, which means Arm NN TfLite Delegate can execute
all TfLite models, and accelerates any operations that Arm NN supports.
In addition, in the delegate options there are some optimizations applied by default in order to improve the inference
performance at the expanse of a slight accuracy reduction. In this example we enable fast math and reduce float32 to
float16 optimizations.

Using the **fast_math** flag can lead to performance improvements in fp32 and fp16 layers but may result in
results with reduced or different precision. The fast_math flag will not have any effect on int8 performance.

The **reduce-fp32-to-fp16** feature works best if all operators of the model are in Fp32. ArmNN will add conversion layers
between layers that weren't in Fp32 in the first place or if the operator is not supported in Fp16.
The overhead of these conversions can lead to a slower overall performance if too many conversions are required.

One can turn off these optimizations in the `create_network` function found in the `network_executor_tflite.py`.
Just change the `optimization_enable` flag to false.

We provide example scripts for performing object detection from video file and video stream with `run_video_file.py` and `run_video_stream.py`.

The application takes a model and video file or camera feed as input, runs inference on each frame, and draws bounding boxes around detected objects, with the corresponding labels and confidence scores overlaid.

A similar implementation of this object detection application is also provided in C++ in the examples for ArmNN.

##### Performing Object Detection with Style Transfer and TensorFlow Lite Delegate
In addition to running Object Detection using TensorFlow Lite Delegate, instead of drawing bounding boxes on each frame, there is an option to run style transfer to create stylized detections.
Style transfer is the ability to create a new image, known as a pastiche, based on two input images: one representing an artistic style and one representing the content frame containing class detections.
The style transfer consists of two submodels:
Style Prediction Model: A MobilenetV2-based neural network that takes an input style image to create a style bottleneck vector.
Style Transform Model: A neural network that applies a style bottleneck vector to a content image and creates a stylized image.
An image containing an art style is preprocessed to a correct size and dimension.
The preprocessed style image is passed to a style predict network which calculates and returns a style bottleneck tensor.
The style transfer network receives the style bottleneck, and a content frame that contains detections, which then transforms the requested class detected and returns a stylized frame.


## Prerequisites

##### PyArmNN

Before proceeding to the next steps, make sure that you have successfully installed the newest version of PyArmNN on your system by following the instructions in the README of the PyArmNN root directory.

You can verify that PyArmNN library is installed and check PyArmNN version using:
```bash
$ pip show pyarmnn
```

You can also verify it by running the following and getting output similar to below:
```bash
$ python -c "import pyarmnn as ann;print(ann.GetVersion())"
'32.0.0'
```

##### Dependencies

Install the following libraries on your system:
```bash
$ sudo apt-get install python3-opencv
```


<b>This section is needed only if running with Arm NN TensorFlow Lite Delegate is desired</b>\
If there is no libarmnnDelegate.so file in your ARMNN_LIB path,
download Arm NN artifacts with Arm NN delegate according to your platform and Arm NN latest version (for this example aarch64 and v21.11 respectively):
```bash
$ export $WORKSPACE=`pwd`
$ mkdir ./armnn_artifacts ; cd armnn_artifacts
$ wget https://github.com/ARM-software/armnn/releases/download/v21.11/ArmNN-linux-aarch64.tar.gz
$ tar -xvzf ArmNN*.tar.gz
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`
```

Create a virtual environment:
```bash
$ python3.7 -m venv devenv --system-site-packages
$ source devenv/bin/activate
```

Install the dependencies from the object_detection example folder:
* In case the python version is 3.8 or lower, tflite_runtime version 2.5.0 (without post1 suffix) should be installed.
  (requirements.txt file should be amended)
```bash
$ cd $WORKSPACE/armnn/python/pyarmnn/examples/object_detection
$ pip install -r requirements.txt
```

---

# Performing Object Detection

## Object Detection from Video File
The `run_video_file.py` example takes a video file as input, runs inference on each frame, and produces frames with bounding boxes drawn around detected objects. The processed frames are written to video file.

The user can specify these arguments at command line:

* `--video_file_path` - <b>Required:</b> Path to the video file to run object detection on

* `--model_file_path` - <b>Required:</b> Path to <b>.tflite, .pb</b> or <b>.onnx</b> object detection model

* `--model_name` - <b>Required:</b> The name of the model being used. Assembles the workflow for the input model. The examples support the model names:

  * `ssd_mobilenet_v1`

  * `yolo_v3_tiny`

* `--label_path` - <b>Required:</b> Path to labels file for the specified model file

* `--output_video_file_path` - Path to the output video file with detections added in

* `--preferred_backends` - You can specify one or more backend in order of preference. Accepted backends include `CpuAcc, GpuAcc, CpuRef`. Arm NN will decide which layers of the network are supported by the backend, falling back to the next if a layer is unsupported. Defaults to `['CpuAcc', 'CpuRef']`

* `--tflite_delegate_path` - Optional. Path to the Arm NN TensorFlow Lite Delegate library (libarmnnDelegate.so). If provided, Arm NN TensorFlow Lite Delegate will be used instead of PyArmNN.

* `--profiling_enabled` - Optional. Enabling this option will print important ML related milestones timing information in micro-seconds. By default, this option is disabled. Accepted options are `true/false`

The `run_video_file.py` example can also perform style transfer on a selected class of detected objects, and stylize the detections based on a given style image.

In addition, to run style transfer, the user needs to specify these arguments at command line:

* `--style_predict_model_file_path` - Path to the style predict model that will be used to create a style bottleneck tensor

* `--style_transfer_model_file_path` - Path to the style transfer model to use which will perform the style transfer

* `--style_image_path` - Path to a .jpg/jpeg/png style image to create stylized frames

* `--style_transfer_class` - A detected class name to transform its style


Run the sample script:
```bash
$ python run_video_file.py --video_file_path <video_file_path> --model_file_path <model_file_path> --model_name <model_name> --tflite_delegate_path <ARMNN delegate file path> --style_predict_model_file_path <style_predict_model_path>
--style_transfer_model_file_path <style_transfer_model_path> --style_image_path <style_image_path> --style_transfer_class <style_transfer_class>
```

## Object Detection from Video Stream
The `run_video_stream.py` example captures frames from a video stream of a device, runs inference on each frame, and produces frames with bounding boxes drawn around detected objects. A window is displayed and refreshed with the latest processed frame.

The user can specify these arguments at command line:

* `--video_source` - Device index to access video stream. Defaults to primary device camera at index 0

* `--model_file_path` - <b>Required:</b> Path to <b>.tflite, .pb</b> or <b>.onnx</b> object detection model

* `--model_name` - <b>Required:</b> The name of the model being used. Assembles the workflow for the input model. The examples support the model names:

  * `ssd_mobilenet_v1`

  * `yolo_v3_tiny`

* `--label_path` - <b>Required:</b> Path to labels file for the specified model file

* `--preferred_backends` - You can specify one or more backend in order of preference. Accepted backends include `CpuAcc, GpuAcc, CpuRef`. Arm NN will decide which layers of the network are supported by the backend, falling back to the next if a layer is unsupported. Defaults to `['CpuAcc', 'CpuRef']`

* `--tflite_delegate_path` - Optional. Path to the Arm NN TensorFlow Lite Delegate library (libarmnnDelegate.so). If provided, Arm NN TensorFlow Lite Delegate will be used instead of PyArmNN.

* `--profiling_enabled` - Optional. Enabling this option will print important ML related milestones timing information in micro-seconds. By default, this option is disabled. Accepted options are `true/false`

Run the sample script:
```bash
$ python run_video_stream.py --model_file_path <model_file_path> --model_name <model_name> --tflite_delegate_path <ARMNN delegate file path> --label_path <Model label path> --video_file_path <Video file>

In addition, to run style trasnfer, the user needs to specify these arguments at command line:

* `--style_predict_model_file_path` - Path to .tflite style predict model that will be used to create a style bottleneck tensor

* `--style_transfer_model_file_path` - Path to .tflite style transfer model to use which will perform the style transfer

* `--style_image_path` - Path to a .jpg/jpeg/png style image to create stylized frames

* `--style_transfer_class` - A detected class name to transform its style

Run the sample script:
```bash
$ python run_video_stream.py --model_file_path <model_file_path> --model_name <model_name> --tflite_delegate_path <ARMNN delegate file path> --style_predict_model_file_path <style_predict_model_path>
--style_transfer_model_file_path <style_transfer_model_path> --style_image_path <style_image_path> --style_transfer_class <style_transfer_class>
```

This application has been verified to work against the MobileNet SSD model and YOLOv3, which can be downloaded along with it's label set from:

* https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip


or from Arm Model Zoo on GitHub.
```bash
sudo apt-get install git git-lfs
git lfs install
git clone https://github.com/arm-software/ml-zoo.git
cd ml-zoo/models/object_detection/yolo_v3_tiny/tflite_fp32/
./get_class_labels.sh
cp labelmappings.txt yolo_v3_tiny_darknet_fp32.tflite $WORKSPACE/armnn/python/pyarmnn/examples/object_detection/
```

The Style Transfer has been verified to work with the following models:

* style prediction model: https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite

* style transfer model: https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite

## Implementing Your Own Network
The examples provide support for `ssd_mobilenet_v1` and `yolo_v3_tiny` models. However, the user is able to add their own network to the object detection scripts by following the steps:

1. Create a new file for your network, for example `network.py`, to contain functions to process the output of the model
2. In that file, the user will need to write a function that decodes the output vectors obtained from running inference on their network and return the bounding box positions of detected objects plus their class index and confidence. Additionally, include a function that returns a resize factor that will scale the obtained bounding boxes to their correct positions in the original frame
3. Import the functions into the main file and, such as with the provided networks, add a conditional statement to the `get_model_processing()` function with the new model name and functions
4. The labels associated with the model can then be passed in with `--label_path` argument

---

# Application Overview

This section provides a walk-through of the application, explaining in detail the steps:

1. Initialisation
2. Creating a Network
3. Preparing the Workload Tensors
4. Executing Inference
5. Postprocessing


### Initialisation

##### Reading from Video Source
After parsing user arguments, the chosen video file or stream is loaded into an OpenCV `cv2.VideoCapture()` object. We use this object to capture frames from the source using the `read()` function.

The `VideoCapture` object also tells us information about the source, such as the frame-rate and resolution of the input video. Using this information, we create a `cv2.VideoWriter()` object which will be used at the end of every loop to write the processed frame to an output video file of the same format as the input.

##### Preparing Labels and Model Specific Functions
In order to interpret the result of running inference on the loaded network, it is required to load the labels associated with the model. In the provided example code, the `dict_labels()` function creates a dictionary that is keyed on the classification index at the output node of the model, with values of the dictionary corresponding to a label and a randomly generated RGB color. This ensures that each class has a unique color which will prove helpful when plotting the bounding boxes of various detected objects in a frame.

Depending on the model being used, the user-specified model name accesses and returns functions to decode and process the inference output, along with a resize factor used when plotting bounding boxes to ensure they are scaled to their correct position in the original frame.


### Creating a Network

##### Creating Parser and Importing Graph
The first step with PyArmNN is to import a graph from file by using the appropriate parser.

The Arm NN SDK provides parsers for reading graphs from a variety of model formats. In our application we specifically focus on `.tflite, .pb, .onnx` models.

Based on the extension of the provided model file, the corresponding parser is created and the network file loaded with `CreateNetworkFromBinaryFile()` function. The parser will handle the creation of the underlying Arm NN graph.

##### Optimizing Graph for Compute Device
Arm NN supports optimized execution on multiple CPU and GPU devices. Prior to executing a graph, we must select the appropriate device context. We do this by creating a runtime context with default options with `IRuntime()`.

We can optimize the imported graph by specifying a list of backends in order of preference and implement backend-specific optimizations. The backends are identified by a string unique to the backend, for example `CpuAcc, GpuAcc, CpuRef`.

Internally and transparently, Arm NN splits the graph into subgraph based on backends, it calls a optimize subgraphs function on each of them and, if possible, substitutes the corresponding subgraph in the original graph with its optimized version.

Using the `Optimize()` function we optimize the graph for inference and load the optimized network onto the compute device with `LoadNetwork()`. This function creates the backend-specific workloads for the layers and a backend specific workload factory which is called to create the workloads.

##### Creating Input and Output Binding Information
Parsers can also be used to extract the input information for the network. By calling `GetSubgraphInputTensorNames` we extract all the input names and, with `GetNetworkInputBindingInfo`, bind the input points of the graph.

The input binding information contains all the essential information about the input. It is a tuple consisting of integer identifiers for bindable layers (inputs, outputs) and the tensor info (data type, quantization information, number of dimensions, total number of elements).

Similarly, we can get the output binding information for an output layer by using the parser to retrieve output tensor names and calling `GetNetworkOutputBindingInfo()`.


### Preparing the Workload Tensors

##### Preprocessing the Captured Frame
Each frame captured from source is read as an `ndarray` in BGR format and therefore has to be preprocessed before being passed into the network.

This preprocessing step consists of swapping channels (BGR to RGB in this example), resizing the frame to the required resolution, expanding dimensions of the array and doing data type conversion to match the model input layer. This information about the input tensor can be readily obtained from reading the `input_binding_info`. For example, SSD MobileNet V1 takes for input a tensor with shape `[1, 300, 300, 3]` and data type `uint8`.

##### Making Input and Output Tensors
To produce the workload tensors, calling the functions `make_input_tensors()` and `make_output_tensors()` will return the input and output tensors respectively.

#### Creating a style bottleneck - Style prediction
If the user decides to use style transfer, a style transfer constructor will be called to create a style bottleneck.
To create a style bottleneck, the style transfer executor will call a style_predict function, which requires a style prediction executor, and an artistic style image.
The style image must be preprocssed to (1, 256, 256, 3) to fit the style predict executor which will then perform inference to create a style bottleneck.

### Executing Inference
After making the workload tensors, a compute device performs inference for the loaded network using the `EnqueueWorkload()` function of the runtime context. By calling the `workload_tensors_to_ndarray()` function, we obtain the results from inference as a list of `ndarrays`.


### Postprocessing

##### Decoding and Processing Inference Output
The output from inference must be decoded to obtain information about detected objects in the frame. In the examples there are implementations for two networks but you may also implement your own network decoding solution here. Please refer to <i>Implementing Your Own Network</i> section of this document to learn how to do this.

For SSD MobileNet V1 models, we decode the results to obtain the bounding box positions, classification index, confidence and number of detections in the input frame.

For YOLO V3 Tiny models, we decode the output and perform non-maximum suppression to filter out any weak detections below a confidence threshold and any redudant bounding boxes above an intersection-over-union threshold.

It is encouraged to experiment with threshold values for confidence and intersection-over-union (IoU) to achieve the best visual results.

The detection results are always returned as a list in the form `[class index, [box positions], confidence score]`, with the box positions list containing bounding box coordinates in the form `[x_min, y_min, x_max, y_max]`.

##### Drawing Bounding Boxes
With the obtained results and using `draw_bounding_boxes()`, we are able to draw bounding boxes around detected objects and add the associated label and confidence score. The labels dictionary created earlier uses the class index of the detected object as a key to return the associated label and color for that class. The resize factor defined at the beginning scales the bounding box coordinates to their correct positions in the original frame. The processed frames are written to file or displayed in a separate window.

##### Creating Stylized Detections
Using the detections, we are able to send them as an input to the style transfer executor to create stylized detections using the style bottleneck tensor that was calculated in the style prediction process.
Each detection will be cropped from the frame, and then preprocessed to (1, 384, 384, 3) to  fit the style transfer executor.
The style transfer executor will use the style bottleneck and the preprocessed content frame to create an artistic stylized frame.
The labels dictionary created earlier uses the class index of the detected object as a key to return the associated label, which is used to identify if it's equal to the style transfer class. The resize factor defined at the beginning scales the bounding box coordinates to their correct positions in the original frame. The processed frames are written to file or displayed in a separate window.
