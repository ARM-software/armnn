#!/usr/bin/env python3
# Copyright © 2020 NXP and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pyarmnn as ann
import numpy as np
import os
from PIL import Image
import example_utils as eu


def preprocess_onnx(img: Image, width: int, height: int, data_type, scale: float, mean: list,
                    stddev: list):
    """Preprocessing function for ONNX imagenet models based on:
    https://github.com/onnx/models/blob/master/vision/classification/imagenet_inference.ipynb

    Args:
        img (PIL.Image): Loaded PIL.Image
        width (int): Target image width
        height (int): Target image height
        data_type: Image datatype (np.uint8 or np.float32)
        scale (float): Scaling factor
        mean: RGB mean values
        stddev: RGB standard deviation

    Returns:
        np.array: Preprocess image as Numpy array
    """
    img = img.resize((256, 256), Image.BILINEAR)
    # first rescale to 256,256 and then center crop
    left = (256 - width) / 2
    top = (256 - height) / 2
    right = (256 + width) / 2
    bottom = (256 + height) / 2
    img = img.crop((left, top, right, bottom))
    img = img.convert('RGB')
    img = np.array(img)
    img = np.reshape(img, (-1, 3))  # reshape to [RGB][RGB]...
    img = ((img / scale) - mean) / stddev
    # NHWC to NCHW conversion, by default NHWC is expected
    # image is loaded as [RGB][RGB][RGB]... transposing it makes it [RRR...][GGG...][BBB...]
    img = np.transpose(img)
    img = img.flatten().astype(data_type)  # flatten into a 1D tensor and convert to float32
    return img


if __name__ == "__main__":
    args = eu.parse_command_line()

    model_filename = 'mobilenetv2-1.0.onnx'
    labels_filename = 'synset.txt'
    archive_filename = 'mobilenetv2-1.0.zip'
    labels_url = 'https://s3.amazonaws.com/onnx-model-zoo/' + labels_filename
    model_url = 'https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/' + model_filename

    # Download resources
    image_filenames = eu.get_images(args.data_dir)

    model_filename, labels_filename = eu.get_model_and_labels(args.model_dir, model_filename, labels_filename,
                                                              archive_filename,
                                                              [model_url, labels_url])

    # all 3 resources must exist to proceed further
    assert os.path.exists(labels_filename)
    assert os.path.exists(model_filename)
    assert image_filenames
    for im in image_filenames:
        assert (os.path.exists(im))

    # Create a network from a model file
    net_id, parser, runtime = eu.create_onnx_network(model_filename)

    # Load input information from the model and create input tensors
    input_binding_info = parser.GetNetworkInputBindingInfo("data")

    # Load output information from the model and create output tensors
    output_binding_info = parser.GetNetworkOutputBindingInfo("mobilenetv20_output_flatten0_reshape0")
    output_tensors = ann.make_output_tensors([output_binding_info])

    # Load labels
    labels = eu.load_labels(labels_filename)

    # Load images and resize to expected size
    images = eu.load_images(image_filenames,
                            224, 224,
                            np.float32,
                            255.0,
                            [0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225],
                            preprocess_onnx)

    eu.run_inference(runtime, net_id, images, labels, input_binding_info, output_binding_info)
