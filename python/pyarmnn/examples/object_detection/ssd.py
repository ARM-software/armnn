# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Contains functions specific to decoding and processing inference results for SSD Mobilenet V1 models.
"""

import cv2
import numpy as np


def ssd_processing(output: np.ndarray, confidence_threshold=0.60):
    """
    Gets class, bounding box positions and confidence from the four outputs of the SSD model.

    Args:
         output: Vector of outputs from network.
         confidence_threshold: Selects only strong detections above this value.

    Returns:
        A list of detected objects in the form [class, [box positions], confidence]
    """
    if len(output) != 4:
        raise RuntimeError('Number of outputs from SSD model does not equal 4')

    position, classification, confidence, num_detections = [index[0] for index in output]

    detections = []
    for i in range(int(num_detections)):
        if confidence[i] > confidence_threshold:
            class_idx = classification[i]
            box = position[i, :4]
            # Reorder positions in format [x_min, y_min, x_max, y_max]
            box[0], box[1], box[2], box[3] = box[1], box[0], box[3], box[2]
            confidence_value = confidence[i]
            detections.append((class_idx, box, confidence_value))
    return detections


def ssd_resize_factor(video: cv2.VideoCapture):
    """
    Gets a multiplier to scale the bounding box positions to
    their correct position in the frame.

    Args:
        video: Video capture object, contains information about data source.

    Returns:
        Resizing factor to scale box coordinates to output frame size.
    """
    frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    return max(frame_height, frame_width)
