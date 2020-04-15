# Copyright 2020 NXP
# SPDX-License-Identifier: MIT

from urllib.parse import urlparse
import os
from PIL import Image
import pyarmnn as ann
import numpy as np
import requests
import argparse
import warnings


def parse_command_line(desc: str = ""):
    """Adds arguments to the script.

    Args:
        desc(str): Script description.

    Returns:
        Namespace: Arguments to the script command.
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-v", "--verbose", help="Increase output verbosity",
                        action="store_true")
    return parser.parse_args()


def __create_network(model_file: str, backends: list, parser=None):
    """Creates a network based on a file and parser type.

    Args:
        model_file (str): Path of the model file.
        backends (list): List of backends to use when running inference.
        parser_type: Parser instance. (pyarmnn.ITFliteParser/pyarmnn.IOnnxParser...)

    Returns:
        int: Network ID.
        int: Graph ID.
        IParser: TF Lite parser instance.
        IRuntime: Runtime object instance.
    """
    args = parse_command_line()
    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    if parser is None:
        # try to determine what parser to create based on model extension
        _, ext = os.path.splitext(model_file)
        if ext == ".onnx":
            parser = ann.IOnnxParser()
        elif ext == ".tflite":
            parser = ann.ITfLiteParser()
    assert (parser is not None)

    network = parser.CreateNetworkFromBinaryFile(model_file)

    preferred_backends = []
    for b in backends:
        preferred_backends.append(ann.BackendId(b))

    opt_network, messages = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(),
                                         ann.OptimizerOptions())
    if args.verbose:
        for m in messages:
            warnings.warn(m)

    net_id, w = runtime.LoadNetwork(opt_network)
    if args.verbose and w:
        warnings.warn(w)

    return net_id, parser, runtime


def create_tflite_network(model_file: str, backends: list = ['CpuAcc', 'CpuRef']):
    """Creates a network from an onnx model file.

    Args:
        model_file (str): Path of the model file.
        backends (list): List of backends to use when running inference.

    Returns:
        int: Network ID.
        int: Graph ID.
        ITFliteParser: TF Lite parser instance.
        IRuntime: Runtime object instance.
    """
    net_id, parser, runtime = __create_network(model_file, backends, ann.ITfLiteParser())
    graph_id = parser.GetSubgraphCount() - 1

    return net_id, graph_id, parser, runtime


def create_onnx_network(model_file: str, backends: list = ['CpuAcc', 'CpuRef']):
    """Creates a network from a tflite model file.

    Args:
        model_file (str): Path of the model file.
        backends (list): List of backends to use when running inference.

    Returns:
        int: Network ID.
        IOnnxParser: ONNX parser instance.
        IRuntime: Runtime object instance.
    """
    return __create_network(model_file, backends, ann.IOnnxParser())


def preprocess_default(img: Image, width: int, height: int, data_type, scale: float, mean: list,
                       stddev: list):
    """Default preprocessing image function.

    Args:
        img (PIL.Image): PIL.Image object instance.
        width (int): Width to resize to.
        height (int): Height to resize to.
        data_type: Data Type to cast the image to.
        scale (float): Scaling value.
        mean (list): RGB mean offset.
        stddev (list): RGB standard deviation.

    Returns:
        np.array: Resized and preprocessed image.
    """
    img = img.resize((width, height), Image.BILINEAR)
    img = img.convert('RGB')
    img = np.array(img)
    img = np.reshape(img, (-1, 3))  # reshape to [RGB][RGB]...
    img = ((img / scale) - mean) / stddev
    img = img.flatten().astype(data_type)
    return img


def load_images(image_files: list, input_width: int, input_height: int, data_type=np.uint8,
                scale: float = 1., mean: list = [0., 0., 0.], stddev: list = [1., 1., 1.],
                preprocess_fn=preprocess_default):
    """Loads images, resizes and performs any additional preprocessing to run inference.

    Args:
        img (list): List of PIL.Image object instances.
        input_width (int): Width to resize to.
        input_height (int): Height to resize to.
        data_type: Data Type to cast the image to.
        scale (float): Scaling value.
        mean (list): RGB mean offset.
        stddev (list): RGB standard deviation.
        preprocess_fn: Preprocessing function.

    Returns:
        np.array: Resized and preprocessed images.
    """
    images = []
    for i in image_files:
        img = Image.open(i)
        img = preprocess_fn(img, input_width, input_height, data_type, scale, mean, stddev)
        images.append(img)
    return images


def load_labels(label_file: str):
    """Loads a labels file containing a label per line.

    Args:
        label_file (str): Labels file path.

    Returns:
        list: List of labels read from a file.
    """
    with open(label_file, 'r') as f:
        labels = [l.rstrip() for l in f]
        return labels
    return None


def print_top_n(N: int, results: list, labels: list, prob: list):
    """Prints TOP-N results

    Args:
        N (int): Result count to print.
        results (list): Top prediction indices.
        labels (list): A list of labels for every class.
        prob (list): A list of probabilities for every class.

    Returns:
        None
    """
    assert (len(results) >= 1 and len(results) == len(labels) == len(prob))
    for i in range(min(len(results), N)):
        print("class={0} ; value={1}".format(labels[results[i]], prob[results[i]]))


def download_file(url: str, force: bool = False, filename: str = None, dest: str = "tmp"):
    """Downloads a file.

    Args:
        url (str): File url.
        force (bool): Forces to download the file even if it exists.
        filename (str): Renames the file when set.

    Returns:
        str: Path to the downloaded file.
    """
    if filename is None:  # extract filename from url when None
        filename = urlparse(url)
        filename = os.path.basename(filename.path)

    if str is not None:
        if not os.path.exists(dest):
            os.makedirs(dest)
        filename = os.path.join(dest, filename)

    print("Downloading '{0}' from '{1}' ...".format(filename, url))
    if not os.path.exists(filename) or force is True:
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)
        print("Finished.")
    else:
        print("File already exists.")

    return filename
