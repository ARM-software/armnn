# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import os

import cv2

from context import network_executor
from context import cv_utils


def test_execute_network(test_data_folder):
    model_path = os.path.join(test_data_folder, "ssd_mobilenet_v1.tflite")
    backends = ["CpuAcc", "CpuRef"]

    executor = network_executor.ArmnnNetworkExecutor(model_path, backends)
    img = cv2.imread(os.path.join(test_data_folder, "messi5.jpg"))
    input_tensors = cv_utils.preprocess(img, executor.input_binding_info, True)

    output_result = executor.run(input_tensors)

    # Ensure it detects a person
    classes = output_result[1]
    assert classes[0][0] == 0
