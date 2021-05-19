# Image Classification with the Arm NN Tensorflow Lite Delegate

This application demonstrates the use of the Arm NN Tensorflow Lite Delegate.
In this application we integrate the Arm NN Tensorflow Lite Delegate into the
TensorFlow Lite Python package.

## Before You Begin

This repository assumes you have built, or have downloaded the
`libarmnnDelegate.so` and `libarmnn.so` from the GitHub releases page. You will
also need to have built the TensorFlow Lite library from source.

If you have not already installed these, please follow our guides in the ArmNN
repository. The guide to build the delegate can be found
[here](../../delegate/BuildGuideNative.md) and the guide to integrate the
delegate into Python can be found
[here](../../delegate/IntegrateDelegateIntoPython.md).


## Getting Started

Before running the application, we will first need to:

- Install the required Python packages
- Download this example
- Download a model and corresponding label mapping
- Download an example image

1. Install required packages and Git Large File Storage (to download models
from the Arm ML-Zoo).

  ```bash
  sudo apt-get install -y python3 python3-pip wget git git-lfs unzip
  git lfs install
  ```

2. Clone the Arm NN repository and change directory to this example.

  ```bash
  git clone https://github.com/arm-software/armnn.git
  cd armnn/samples/ImageClassification
  ```

3. Download your model and label mappings.

  For this example we use the `MobileNetV2` model. This model can be found in
  the Arm ML-Zoo as well as scripts to download the labels for the model.

  ```bash
  export BASEDIR=$(pwd)
  #clone the model zoo
  git clone https://github.com/arm-software/ml-zoo.git
  #go to the mobilenetv2 uint8 folder
  cd ml-zoo/models/image_classification/mobilenet_v2_1.0_224/tflite_uint8
  #generate the labelmapping
  ./get_class_labels.sh
  #cd back to this project folder
  cd BASEDIR
  #copy your model and label mapping
  cp ml-zoo/models/image_classification/mobilenet_v2_1.0_224/tflite_uint8/mobilenet_v2_1.0_224_quantized_1_default_1.tflite .
  cp ml-zoo/models/image_classification/mobilenet_v2_1.0_224/tflite_uint8 labelmappings.txt .
  ```

4. Download a test image.

  ```bash
  wget -O cat.png "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
  ```

5. Download the required Python packages.

  ```bash
  pip3 install -r requirements.txt
  ```

6. Copy over your `libtensorflow_lite_all.so` and `libarmnn.so` library files
you built/downloaded before trying this application to the application
folder. For example:

  ```bash
  cp path/to/tensorflow/directory/tensorflow/bazel-bin/libtensorflow_lite_all.so .
  cp /path/to/armnn/binaries/libarmnn.so .
  ```

## Folder Structure

You should now have the following folder structure:

```
.
├── README.md
├── run_classifier.py          # script for the demo
├── libtensorflow_lite_all.so  # tflite library built from tensorflow
├── libarmnn.so
├── cat.png                    # downloaded example image
├── mobilenet_v2_1.0_224_quantized_1_default_1.tflite #tflite model from ml-zoo
└── labelmappings.txt          # model labelmappings for output processing
```

## Run the model

```bash
python3 run_classifier.py \
--input_image cat.png \
--model_file mobilenet_v2_1.0_224_quantized_1_default_1.tflite \
--label_file labelmappings.txt \
--delegate_path /path/to/delegate/libarmnnDelegate.so.24 \
--preferred_backends GpuAcc CpuAcc CpuRef
```

The output prediction will be printed. In this example we get:

```bash
'tabby, tabby cat'
```

## Running an inference with the Arm NN TensorFlow Lite Delegate

Compared to your usual TensorFlow Lite projects, using the Arm NN TensorFlow
Lite Delegate requires one extra step when loading in your model:

```python
import tflite_runtime.interpreter as tflite

armnn_delegate = tflite.load_delegate("/path/to/delegate/libarmnnDelegate.so",
  options={
    "backends": "GpuAcc,CpuAcc,CpuRef",
    "logging-severity": "info"
  }
)
interpreter = tflite.Interpreter(
  model_path="mobilenet_v2_1.0_224_quantized_1_default_1.tflite",
  experimental_delegates=[armnn_delegate]
)
```
