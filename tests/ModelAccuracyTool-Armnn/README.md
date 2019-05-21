# The ModelAccuracyTool-Armnn

The `ModelAccuracyTool-Armnn` is a program for measuring the Top 5 accuracy results of a model against an image dataset.

Prerequisites:
1. The model is in .armnn format model file. The `ArmnnConverter` can be used to convert a model to this format.
2. The ImageNet test data is in raw tensor file format. The `ImageTensorGenerator` can be used to convert the test
images to this format.

|Cmd:|||
| ---|---|---|
| -h | --help                   | Display help messages |
| -m | --model-path             | Path to armnn format model file |
| -c | --compute                | Which device to run layers on by default. Possible choices: CpuRef, CpuAcc, GpuAcc. Default: CpuAcc, CpuRef |
| -d | --data-dir               | Path to directory containing the ImageNet test data |
| -i | --input-name             | Identifier of the input tensors in the network separated by comma |
| -o | --output-name            | Identifier of the output tensors in the network separated by comma |
| -v | --validation-labels-path | Path to ImageNet Validation Label file |

Example usage: <br>
<code>./ModelAccuracyTool -m /path/to/model/model.armnn -c CpuRef -d /path/to/test/directory/ -i input -o output
-v /path/to/file/val.txt</code>
