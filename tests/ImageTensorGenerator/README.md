# The ImageTensorGenerator

The `ImageTensorGenerator` is a program for generating a .raw tensor file from a .jpg image.

|Cmd:|||
| ---|---|---|
| -h | --help    | Display help messages |
| -i | --infile  | Input image file to generate tensor from |
| -l | --layout  | Output data layout, "NHWC" or "NCHW". Default value: NHWC |
| -o | --outfile | Output raw tensor file path |

Example usage: <br>
<code>./ImageTensorGenerator -i /path/to/image/dog.jpg -l NHWC -o /output/path/dog.raw</code>
