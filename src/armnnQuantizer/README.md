# The ArmnnQuantizer

The `ArmnnQuantizer` is a program for loading a 32-bit float network into ArmNN and converting it into a quantized asymmetric 8-bit or quantized symmetric 16-bit network.
It supports static quantization by default, dynamic quantization is enabled if CSV file of raw input tensors is provided. Run the program with no arguments to see command-line help.


|Cmd:|||
| ---|---|---|
| -h | --help               | Display help messages |
| -f | --infile             | Input file containing float 32 ArmNN Input Graph |
| -s | --scheme             | Quantization scheme, "QAsymm8" or "QSymm16". Default value: QAsymm8 |
| -c | --csvfile            | CSV file containing paths for raw input tensors for dynamic quantization. If unset, static quantization is used |
| -p | --preserve-data-type | Preserve the input and output data types. If unset, input and output data types are not preserved |
| -d | --outdir             | Directory that output file will be written to |
| -o | --outfile            | ArmNN output file name |

Example usage: <br>
<code>./ArmnnQuantizer -f /path/to/armnn/input/graph/ -s "QSymm16" -c /path/to/csv/file -p 1 -d /path/to/output -o outputFileName</code>
