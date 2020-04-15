# Copyright 2020 NXP
# SPDX-License-Identifier: MIT

from zipfile import ZipFile
import numpy as np
import pyarmnn as ann
import example_utils as eu
import os


def unzip_file(filename):
    """Unzips a file to its current location.

    Args:
        filename (str): Name of the archive.

    Returns:
        str: Directory path of the extracted files.
    """
    with ZipFile(filename, 'r') as zip_obj:
        zip_obj.extractall(os.path.dirname(filename))
    return os.path.dirname(filename)


if __name__ == "__main__":
    # Download resources
    archive_filename = eu.download_file(
        'https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip')
    dir_path = unzip_file(archive_filename)
    # names of the files in the archive
    labels_filename = os.path.join(dir_path, 'labels_mobilenet_quant_v1_224.txt')
    model_filename = os.path.join(dir_path, 'mobilenet_v1_1.0_224_quant.tflite')
    kitten_filename = eu.download_file('https://s3.amazonaws.com/model-server/inputs/kitten.jpg')

    # Create a network from the model file
    net_id, graph_id, parser, runtime = eu.create_tflite_network(model_filename)

    # Load input information from the model
    # tflite has all the need information in the model unlike other formats
    input_names = parser.GetSubgraphInputTensorNames(graph_id)
    assert len(input_names) == 1  # there should be 1 input tensor in mobilenet

    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
    input_width = input_binding_info[1].GetShape()[1]
    input_height = input_binding_info[1].GetShape()[2]

    # Load output information from the model and create output tensors
    output_names = parser.GetSubgraphOutputTensorNames(graph_id)
    assert len(output_names) == 1  # and only one output tensor
    output_binding_info = parser.GetNetworkOutputBindingInfo(graph_id, output_names[0])
    output_tensors = ann.make_output_tensors([output_binding_info])

    # Load labels file
    labels = eu.load_labels(labels_filename)

    # Load images and resize to expected size
    image_names = [kitten_filename]
    images = eu.load_images(image_names, input_width, input_height)

    for idx, im in enumerate(images):
        # Create input tensors
        input_tensors = ann.make_input_tensors([input_binding_info], [im])

        # Run inference
        print("Running inference on '{0}' ...".format(image_names[idx]))
        runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

        # Process output
        out_tensor = ann.workload_tensors_to_ndarray(output_tensors)[0][0]
        results = np.argsort(out_tensor)[::-1]
        eu.print_top_n(5, results, labels, out_tensor)
