# The ImageCSVFileGenerator

The `ImageCSVFileGenerator` is a program for creating a CSV file that contains a list of .raw tensor files. These
.raw tensor files can be generated using the`ImageTensorGenerator`.

|Cmd:|||
| ---|---|---|
| -h | --help    | Display help messages |
| -i | --indir   | Directory that .raw files are stored in |
| -o | --outfile | Output CSV file path |

Example usage: <br>
<code>./ImageCSVFileGenerator -i /path/to/directory/ -o /output/path/csvfile.csv</code>
