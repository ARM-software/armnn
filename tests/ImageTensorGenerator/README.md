# The ImageTensorGenerator

The `ImageTensorGenerator` is a program for pre-processing a .jpg image before generating a .raw tensor file from it.

Build option:
To build ModelAccuracyTool, pass the following options to Cmake:
* -DBUILD_ACCURACY_TOOL=1

|Cmd:|||
| ---|---|---|
| -h | --help         | Display help messages |
| -f | --model-format | Format of the intended model file that uses the images.Different formats have different image normalization styles.Accepted values (tflite) |
| -i | --infile       | Input image file to generate tensor from |
| -o | --outfile      | Output raw tensor file path |
| -z | --output-type  | The data type of the output tensors.If unset, defaults to "float" for all defined inputs. Accepted values (float, int or qasymm8)
|    | --new-width    |Resize image to new width. Keep original width if unspecified |
|    | --new-height   |             Resize image to new height. Keep original height if unspecified |
| -l | --layout       | Output data layout, "NHWC" or "NCHW". Default value: NHWC |

Example usage: <br>
<code>./ImageTensorGenerator -i /path/to/image/dog.jpg -o /output/path/dog.raw --new-width 224 --new-height 224</code>