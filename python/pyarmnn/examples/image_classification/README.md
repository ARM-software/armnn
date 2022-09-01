# PyArmNN Image Classification Sample Application

## Overview

To further explore PyArmNN API, we provide an example for running image classification on an image.

All resources are downloaded during execution, so if you do not have access to the internet, you may need to download these manually. The file `example_utils.py` contains code shared between the examples.

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

Install the dependencies:

```bash
$ pip install -r requirements.txt
```

## Perform Image Classification

Perform inference with TFLite model by running the sample script:
```bash
$ python tflite_mobilenetv1_quantized.py
```

Perform inference with ONNX model by running the sample script:
```bash
$ python onnx_mobilenetv2.py
```

The output from inference will be printed as <i>Top N</i> results, listing the classes and probabilities associated with the classified image.
