#!/usr/bin/env python3
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT

import numpy as np
import pyarmnn as ann
import example_utils as eu
import os

args = eu.parse_command_line()

# names of the files in the archive
labels_filename = 'labels_mobilenet_quant_v1_224.txt'
model_filename = 'mobilenet_v1_1.0_224_quant.tflite'
archive_filename = 'mobilenet_v1_1.0_224_quant_and_labels.zip'

archive_url = \
    'https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip'

model_filename, labels_filename = eu.get_model_and_labels(args.model_dir, model_filename, labels_filename,
                                                          archive_filename, archive_url)

image_filenames = eu.get_images(args.data_dir)

# all 3 resources must exist to proceed further
assert os.path.exists(labels_filename)
assert os.path.exists(model_filename)
assert image_filenames
for im in image_filenames:
    assert(os.path.exists(im))

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

# Load labels file
labels = eu.load_labels(labels_filename)

# Load images and resize to expected size
images = eu.load_images(image_filenames, input_width, input_height)

eu.run_inference(runtime, net_id, images, labels, input_binding_info, output_binding_info)
