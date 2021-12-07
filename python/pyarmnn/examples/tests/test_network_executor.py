# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import pytest
import cv2
import numpy as np

from context import network_executor
from context import network_executor_tflite
from context import cv_utils

@pytest.mark.parametrize("executor_name", ["armnn", "tflite"])
def test_execute_network(test_data_folder, executor_name):
    model_path = os.path.join(test_data_folder, "ssd_mobilenet_v1.tflite")
    backends = ["CpuAcc", "CpuRef"]
    if executor_name == "armnn":
        executor = network_executor.ArmnnNetworkExecutor(model_path, backends)
    elif executor_name == "tflite":
        delegate_path = os.path.join(test_data_folder, "libarmnnDelegate.so")
        executor = network_executor_tflite.TFLiteNetworkExecutor(model_path, backends, delegate_path)
    else:
        raise f"unsupported executor_name: {executor_name}"

    img = cv2.imread(os.path.join(test_data_folder, "messi5.jpg"))
    resized_img = cv_utils.preprocess(img, executor.get_data_type(), executor.get_shape(), True)

    output_result = executor.run([resized_img])

    # Ensure it detects a person
    classes = output_result[1]
    assert classes[0][0] == 0

    # Unit tests for network executor class functions - specifically for ssd_mobilenet_v1.tflite network
    assert executor.get_data_type() == np.uint8
    assert executor.get_shape() == (1, 300, 300, 3)
