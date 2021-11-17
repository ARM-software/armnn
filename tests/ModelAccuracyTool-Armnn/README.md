# The ModelAccuracyTool-Armnn

The `ModelAccuracyTool-Armnn` is a program for measuring the Top 5 accuracy results of a model against an image dataset.

Prerequisites:
1. The model is in .armnn format model file. The `ArmnnConverter` can be used to convert a model to this format.

Build option:
To build ModelAccuracyTool, pass the following options to Cmake:
* -DFLATC_DIR=/path/to/flatbuffers/x86build/
* -DBUILD_ACCURACY_TOOL=1
* -DBUILD_ARMNN_SERIALIZER=1

|Cmd:|||
| ---|---|---|
| -h | --help                   | Display help messages |
| -m | --model-path             | Path to armnn format model file |
| -f | --model-format           | The model format. Supported values: tflite |
| -i | --input-name             | Identifier of the input tensors in the network separated by comma |
| -o | --output-name            | Identifier of the output tensors in the network separated by comma |
| -d | --data-dir               | Path to directory containing the ImageNet test data |
| -p | --model-output-labels    | Path to model output labels file.
| -v | --validation-labels-path | Path to ImageNet Validation Label file
| -l | --data-layout ]          | Data layout. Supported value: NHWC, NCHW. Default: NHWC
| -c | --compute                | Which device to run layers on by default. Possible choices: CpuRef, CpuAcc, GpuAcc. Default: CpuAcc, CpuRef |
| -r | --validation-range       | The range of the images to be evaluated. Specified in the form <begin index>:<end index>. The index starts at 1 and the range is inclusive. By default the evaluation will be performed on all images. |
| -e | --excludelist-path       | Path to a excludelist file where each line denotes the index of an image to be excluded from evaluation. |

Example usage: <br>
<code>./ModelAccuracyTool -m /path/to/model/model.armnn -f tflite -i input -o output -d /path/to/test/directory/ -p /path/to/model-output-labels -v /path/to/file/val.txt -c CpuRef -r 1:100</code>