# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import cv2
import numpy as np

from context import style_transfer
from context import cv_utils


def test_style_transfer_postprocess(test_data_folder):
    content_image = "messi5.jpg"
    target_shape = (1,256,256,3)
    keep_aspect_ratio = False
    image = cv2.imread(os.path.join(test_data_folder, content_image))
    original_shape = image.shape
    preprocessed_image = cv_utils.preprocess(image, np.float32, target_shape, False, keep_aspect_ratio)
    assert preprocessed_image.shape == target_shape

    postprocess_image = style_transfer.style_transfer_postprocess(preprocessed_image, original_shape)
    assert postprocess_image.shape == original_shape


def test_style_transfer(test_data_folder):
    style_predict_model_path = os.path.join(test_data_folder, "style_predict.tflite")
    style_transfer_model_path = os.path.join(test_data_folder, "style_transfer.tflite")
    backends = ["CpuAcc", "CpuRef"]
    delegate_path = os.path.join(test_data_folder, "libarmnnDelegate.so")
    image = cv2.imread(os.path.join(test_data_folder, "messi5.jpg"))

    style_transfer_executor = style_transfer.StyleTransfer(style_predict_model_path, style_transfer_model_path,
                                                           image, backends, delegate_path)

    assert style_transfer_executor.get_style_predict_executor_shape() == (1, 256, 256, 3)

def test_run_style_transfer(test_data_folder):
    style_predict_model_path = os.path.join(test_data_folder, "style_predict.tflite")
    style_transfer_model_path = os.path.join(test_data_folder, "style_transfer.tflite")
    backends = ["CpuAcc", "CpuRef"]
    delegate_path = os.path.join(test_data_folder, "libarmnnDelegate.so")
    style_image = cv2.imread(os.path.join(test_data_folder, "messi5.jpg"))
    content_image = cv2.imread(os.path.join(test_data_folder, "basketball1.png"))

    style_transfer_executor = style_transfer.StyleTransfer(style_predict_model_path, style_transfer_model_path,
                                                           style_image, backends, delegate_path)

    stylized_image = style_transfer_executor.run_style_transfer(content_image)
    assert stylized_image.shape == content_image.shape


def test_create_stylized_detection(test_data_folder):
    style_predict_model_path = os.path.join(test_data_folder, "style_predict.tflite")
    style_transfer_model_path = os.path.join(test_data_folder, "style_transfer.tflite")
    backends = ["CpuAcc", "CpuRef"]
    delegate_path = os.path.join(test_data_folder, "libarmnnDelegate.so")

    style_image = cv2.imread(os.path.join(test_data_folder, "messi5.jpg"))
    content_image = cv2.imread(os.path.join(test_data_folder, "basketball1.png"))
    detections = [(0.0, [0.16745174, 0.15101701, 0.5371381, 0.74165875], 0.87597656)]
    labels = {0: ('person', (50.888902345757494, 129.61878417939724, 207.2891028294508)),
             1: ('bicycle', (55.055339686943654, 55.828708219750574, 43.550389695374676)),
             2: ('car', (95.33096265662336, 194.872841553212, 218.58516479057758))}
    style_transfer_executor = style_transfer.StyleTransfer(style_predict_model_path, style_transfer_model_path,
                                                           style_image, backends, delegate_path)

    stylized_image = style_transfer.create_stylized_detection(style_transfer_executor, 'person', content_image,
                                                              detections, 720, labels)

    assert stylized_image.shape == content_image.shape
