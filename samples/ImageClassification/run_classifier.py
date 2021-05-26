import argparse
from pathlib import Path
from typing import Union

import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np


def check_args(args: argparse.Namespace):
    """Check the values used in the command-line have acceptable values

    args:
      - args: argparse.Namespace

    returns:
      - None

    raises:
      - FileNotFoundError: if passed files do not exist.
      - IOError: if files are of incorrect format.
    """
    input_image_p = args.input_image
    if not input_image_p.suffix in (".png", ".jpg", ".jpeg"):
        raise IOError(
            "--input_image option should point to an image file of the "
            "format .jpg, .jpeg, .png"
        )
    if not input_image_p.exists():
        raise FileNotFoundError("Cannot find ", input_image_p.name)
    model_p = args.model_file
    if not model_p.suffix == ".tflite":
        raise IOError("--model_file should point to a tflite file.")
    if not model_p.exists():
        raise FileNotFoundError("Cannot find ", model_p.name)
    label_mapping_p = args.label_file
    if not label_mapping_p.suffix == ".txt":
        raise IOError("--label_file expects a .txt file.")
    if not label_mapping_p.exists():
        raise FileNotFoundError("Cannot find ", label_mapping_p.name)

    # check all args given in preferred backends make sense
    supported_backends = ["GpuAcc", "CpuAcc", "CpuRef"]
    if not all([backend in supported_backends for backend in args.preferred_backends]):
        raise ValueError("Incorrect backends given. Please choose from "\
            "'GpuAcc', 'CpuAcc', 'CpuRef'.")

    return None


def load_image(image_path: Path, model_input_dims: Union[tuple, list], grayscale: bool):
    """load an image and put into correct format for the tensorflow lite model

    args:
      - image_path: pathlib.Path
      - model_input_dims: tuple (or array-like). (height,width)

    returns:
      - image: np.array
    """
    height, width = model_input_dims
    # load and resize image
    image = Image.open(image_path).resize((width, height))
    # convert to greyscale if expected
    if grayscale:
        image = image.convert("LA")

    image = np.expand_dims(image, axis=0)

    return image


def load_delegate(delegate_path: Path, backends: list):
    """load the armnn delegate.

    args:
      - delegate_path: pathlib.Path -> location of you libarmnnDelegate.so
      - backends: list -> list of backends you want to use in string format

    returns:
      - armnn_delegate: tflite.delegate
    """
    # create a command separated string
    backend_string = ",".join(backends)
    # load delegate
    armnn_delegate = tflite.load_delegate(
        library=delegate_path,
        options={"backends": backend_string, "logging-severity": "info"},
    )

    return armnn_delegate


def load_tf_model(model_path: Path, armnn_delegate: tflite.Delegate):
    """load a tflite model for use with the armnn delegate.

    args:
      - model_path: pathlib.Path
      - armnn_delegate: tflite.TfLiteDelegate

    returns:
      - interpreter: tflite.Interpreter
    """
    interpreter = tflite.Interpreter(
        model_path=model_path.as_posix(), experimental_delegates=[armnn_delegate]
    )
    interpreter.allocate_tensors()

    return interpreter


def run_inference(interpreter, input_image):
    """Run inference on a processed input image and return the output from
    inference.

    args:
      - interpreter: tflite_runtime.interpreter.Interpreter
      - input_image: np.array

    returns:
      - output_data: np.array
    """
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Test model on random input data.
    interpreter.set_tensor(input_details[0]["index"], input_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    return output_data


def create_mapping(label_mapping_p):
    """Creates a Python dictionary mapping an index to a label.

    label_mapping[idx] = label

    args:
      - label_mapping_p: pathlib.Path

    returns:
      - label_mapping: dict
    """
    idx = 0
    label_mapping = {}
    with open(label_mapping_p) as label_mapping_raw:
        for line in label_mapping_raw:
            label_mapping[idx] = line
            idx += 1

    return label_mapping


def process_output(output_data, label_mapping):
    """Process the output tensor into a label from the labelmapping file. Takes
    the index of the maximum valur from the output array.

    args:
      - output_data: np.array
      - label_mapping: dict

    returns:
      - str: labelmapping for max index.
    """
    idx = np.argmax(output_data[0])

    return label_mapping[idx]


def main(args):
    """Run the inference for options passed in the command line.

    args:
      - args: argparse.Namespace

    returns:
      - None
    """
    # sanity check on args
    check_args(args)
    # load in the armnn delegate
    armnn_delegate = load_delegate(args.delegate_path, args.preferred_backends)
    # load tflite model
    interpreter = load_tf_model(args.model_file, armnn_delegate)
    # get input shape for image resizing
    input_shape = interpreter.get_input_details()[0]["shape"]
    height, width = input_shape[1], input_shape[2]
    input_shape = (height, width)
    # load input image
    input_image = load_image(args.input_image, input_shape, False)
    # get label mapping
    labelmapping = create_mapping(args.label_file)
    output_tensor = run_inference(interpreter, input_image)
    output_prediction = process_output(output_tensor, labelmapping)

    print("Prediction: ", output_prediction)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_image", help="File path of image file", type=Path, required=True
    )
    parser.add_argument(
        "--model_file",
        help="File path of the model tflite file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--label_file",
        help="File path of model labelmapping file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--delegate_path",
        help="File path of ArmNN delegate file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--preferred_backends",
        help="list of backends in order of preference",
        type=str,
        nargs="+",
        required=False,
        default=["CpuAcc", "CpuRef"],
    )
    args = parser.parse_args()

    main(args)
