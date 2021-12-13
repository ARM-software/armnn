# PyArmNN Object Detection Sample Application

## Introduction
This sample application guides the user and shows how to perform object detection using PyArmNN API. We assume the user has already built PyArmNN by following the instructions of the README in the main PyArmNN directory.

We provide example scripts for performing object detection from video file and video stream with `run_video_file.py` and `run_video_stream.py`.

The application takes a model and video file or camera feed as input, runs inference on each frame, and draws bounding boxes around detected objects, with the corresponding labels and confidence scores overlaid.

A similar implementation of this object detection application is also provided in C++ in the examples for ArmNN.

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
'28.0.0'
```

##### Dependencies

Install the following libraries on your system:
```bash
$ sudo apt-get install python3-opencv libqtgui4 libqt4-test
```

Create a virtual environment:
```bash
$ python3.7 -m venv devenv --system-site-packages
$ source devenv/bin/activate
```

Install the dependencies:
```bash
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


Run the sample script:
```bash
$ python run_video_file.py --video_file_path <video_file_path> --model_file_path <model_file_path> --model_name <model_name>
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


Run the sample script:
```bash
$ python run_video_stream.py --model_file_path <model_file_path> --model_name <model_name>
```

This application has been verified to work against the MobileNet SSD model, which can be downloaded along with it's label set from:

* https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip

## Implementing Your Own Network
The examples provide support for `ssd_mobilenet_v1` and `yolo_v3_tiny` models. However, the user is able to add their own network to the object detection scripts by following the steps:

1. Create a new file for your network, for example `network.py`, to contain functions to process the output of the model
2. In that file, the user will need to write a function that decodes the output vectors obtained from running inference on their network and return the bounding box positions of detected objects plus their class index and confidence. Additionally, include a function that returns a resize factor that will scale the obtained bounding boxes to their correct positions in the original frame
3. Import the functions into the main file and, such as with the provided networks, add a conditional statement to the `get_model_processing()` function with the new model name and functions
4. The labels associated with the model can then be passed in with `--label_path` argument

---

# Application Overview

This section provides a walkthrough of the application, explaining in detail the steps:

1. Initialisation
2. Creating a Network
3. Preparing the Workload Tensors
4. Executing Inference
5. Postprocessing


### Initialisation

##### Reading from Video Source
After parsing user arguments, the chosen video file or stream is loaded into an OpenCV `cv2.VideoCapture()` object. We use this object to capture frames from the source using the `read()` function.

The `VideoCapture` object also tells us information about the source, such as the framerate and resolution of the input video. Using this information, we create a `cv2.VideoWriter()` object which will be used at the end of every loop to write the processed frame to an output video file of the same format as the input.

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
