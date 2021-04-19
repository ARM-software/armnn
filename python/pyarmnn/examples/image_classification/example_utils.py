# Copyright © 2020 NXP and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

from urllib.parse import urlparse
from PIL import Image
from zipfile import ZipFile
import os
import pyarmnn as ann
import numpy as np
import requests
import argparse
import warnings

DEFAULT_IMAGE_URL = 'https://s3.amazonaws.com/model-server/inputs/kitten.jpg'


def run_inference(runtime, net_id, images, labels, input_binding_info, output_binding_info):
    """Runs inference on a set of images.

    Args:
        runtime: Arm NN runtime
        net_id: Network ID
        images: Loaded images to run inference on
        labels: Loaded labels per class
        input_binding_info: Network input information
        output_binding_info: Network output information

    Returns:
        None
    """
    output_tensors = ann.make_output_tensors([output_binding_info])
    for idx, im in enumerate(images):
        # Create input tensors
        input_tensors = ann.make_input_tensors([input_binding_info], [im])

        # Run inference
        print("Running inference({0}) ...".format(idx))
        runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

        # Process output
        # output tensor has a shape (1, 1001)
        out_tensor = ann.workload_tensors_to_ndarray(output_tensors)[0][0]
        results = np.argsort(out_tensor)[::-1]
        print_top_n(5, results, labels, out_tensor)


def unzip_file(filename: str):
    """Unzips a file.

    Args:
        filename(str): Name of the file

    Returns:
        None
    """
    with ZipFile(filename, 'r') as zip_obj:
        zip_obj.extractall()


def parse_command_line(desc: str = ""):
    """Adds arguments to the script.

    Args:
        desc (str): Script description

    Returns:
        Namespace: Arguments to the script command
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-v", "--verbose", help="Increase output verbosity",
                        action="store_true")
    parser.add_argument("-d", "--data-dir", help="Data directory which contains all the images.",
                        action="store", default="")
    parser.add_argument("-m", "--model-dir",
                        help="Model directory which contains the model file (TFLite, ONNX).", action="store",
                        default="")
    return parser.parse_args()


def __create_network(model_file: str, backends: list, parser=None):
    """Creates a network based on a file and parser type.

    Args:
        model_file (str): Path of the model file
        backends (list): List of backends to use when running inference.
        parser_type: Parser instance. (pyarmnn.ITFliteParser/pyarmnn.IOnnxParser...)

    Returns:
        int: Network ID
        IParser: TF Lite parser instance
        IRuntime: Runtime object instance
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


def create_tflite_network(model_file: str, backends: list = ('CpuAcc', 'CpuRef')):
    """Creates a network from a tflite model file.

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


def create_onnx_network(model_file: str, backends: list = ('CpuAcc', 'CpuRef')):
    """Creates a network from an onnx model file.

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
                scale: float = 1., mean: list = (0., 0., 0.), stddev: list = (1., 1., 1.),
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


def download_file(url: str, force: bool = False, filename: str = None):
    """Downloads a file.

    Args:
        url (str): File url.
        force (bool): Forces to download the file even if it exists.
        filename (str): Renames the file when set.

    Raises:
        RuntimeError: If for some reason download fails.

    Returns:
        str: Path to the downloaded file.
    """
    try:
        if filename is None:  # extract filename from url when None
            filename = urlparse(url)
            filename = os.path.basename(filename.path)

        print("Downloading '{0}' from '{1}' ...".format(filename, url))
        if not os.path.exists(filename) or force is True:
            r = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(r.content)
            print("Finished.")
        else:
            print("File already exists.")
    except:
        raise RuntimeError("Unable to download file.")

    return filename


def get_model_and_labels(model_dir: str, model: str, labels: str, archive: str = None, download_url: str = None):
    """Gets model and labels.

    Args:
        model_dir(str): Folder in which model and label files can be found
        model (str): Name of the model file
        labels (str): Name of the labels file
        archive (str): Name of the archive file (optional - need to provide only labels and model)
        download_url(str or list): Archive url or urls if multiple files (optional - to to provide only to download it)

    Returns:
        tuple (str, str): Output label and model filenames
    """
    labels = os.path.join(model_dir, labels)
    model = os.path.join(model_dir, model)

    if os.path.exists(labels) and os.path.exists(model):
        print("Found model ({0}) and labels ({1}).".format(model, labels))
    elif archive is not None and os.path.exists(os.path.join(model_dir, archive)):
        print("Found archive ({0}). Unzipping ...".format(archive))
        unzip_file(archive)
    elif download_url is not None:
        print("Model, labels or archive not found. Downloading ...".format(archive))
        try:
            if isinstance(download_url, str):
                download_url = [download_url]
            for dl in download_url:
                archive = download_file(dl)
                if dl.lower().endswith(".zip"):
                    unzip_file(archive)
        except RuntimeError:
            print("Unable to download file ({}).".format(download_url))

    if not os.path.exists(labels) or not os.path.exists(model):
        raise RuntimeError("Unable to provide model and labels.")

    return model, labels


def list_images(folder: str = None, formats: list = ('.jpg', '.jpeg')):
    """Lists files of a certain format in a folder.

    Args:
        folder (str): Path to the folder to search
        formats (list): List of supported files

    Returns:
        list: A list of found files
    """
    files = []
    if folder and not os.path.exists(folder):
        print("Folder '{}' does not exist.".format(folder))
        return files

    for file in os.listdir(folder if folder else os.getcwd()):
        for frmt in formats:
            if file.lower().endswith(frmt):
                files.append(os.path.join(folder, file) if folder else file)
                break  # only the format loop

    return files


def get_images(image_dir: str, image_url: str = DEFAULT_IMAGE_URL):
    """Gets image.

    Args:
        image_dir (str): Image filename
        image_url (str): Image url

    Returns:
        str: Output image filename
    """
    images = list_images(image_dir)
    if not images and image_url is not None:
        print("No images found. Downloading ...")
        try:
            images = [download_file(image_url)]
        except RuntimeError:
            print("Unable to download file ({0}).".format(image_url))

    if not images:
        raise RuntimeError("Unable to provide images.")

    return images
