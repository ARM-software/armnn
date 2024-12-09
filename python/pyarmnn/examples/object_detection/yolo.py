# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Contains functions specific to decoding and processing inference results for YOLO V3 Tiny models.
"""

import cv2
import numpy as np


def iou(box1: list, box2: list):
    """
    Calculates the intersection-over-union (IoU) value for two bounding boxes.

    Args:
        box1: Array of positions for first bounding box
              in the form [x_min, y_min, x_max, y_max].
        box2: Array of positions for second bounding box.

    Returns:
        Calculated intersection-over-union (IoU) value for two bounding boxes.
    """
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if area_box1 <= 0 or area_box2 <= 0:
        iou_value = 0
    else:
        y_min_intersection = max(box1[1], box2[1])
        x_min_intersection = max(box1[0], box2[0])
        y_max_intersection = min(box1[3], box2[3])
        x_max_intersection = min(box1[2], box2[2])

        area_intersection = max(0, y_max_intersection - y_min_intersection) *\
                            max(0, x_max_intersection - x_min_intersection)
        area_union = area_box1 + area_box2 - area_intersection

        try:
            iou_value = area_intersection / area_union
        except ZeroDivisionError:
            iou_value = 0

    return iou_value


def yolo_processing(output: np.ndarray, confidence_threshold=0.40, iou_threshold=0.40):
    """
    Performs non-maximum suppression on input detections. Any detections
    with IOU value greater than given threshold are suppressed.

    Args:
        output: Vector of outputs from network.
        confidence_threshold: Selects only strong detections above this value.
        iou_threshold: Filters out boxes with IOU values above this value.

    Returns:
        A list of detected objects in the form [class, [box positions], confidence]
    """
    if len(output) != 1:
        raise RuntimeError('Number of outputs from YOLO model does not equal 1')

    # Find the array index of detections with confidence value above threshold
    confidence_det = output[0][:, :, 4][0]
    detections = list(np.where(confidence_det > confidence_threshold)[0])
    all_det, nms_det = [], []

    # Create list of all detections above confidence threshold
    for d in detections:
        box_positions = list(output[0][:, d, :4][0])
        confidence_score = output[0][:, d, 4][0]
        class_idx = np.argmax(output[0][:, d, 5:])
        all_det.append((class_idx, box_positions, confidence_score))

    # Suppress detections with IOU value above threshold
    while all_det:
        element = int(np.argmax([all_det[i][2] for i in range(len(all_det))]))
        nms_det.append(all_det.pop(element))
        all_det = [*filter(lambda x: (iou(x[1], nms_det[-1][1]) <= iou_threshold), [det for det in all_det])]
    return nms_det


def yolo_resize_factor(video: cv2.VideoCapture, input_data_shape: tuple):
    """
    Gets a multiplier to scale the bounding box positions to
    their correct position in the frame.

    Args:
        video: Video capture object, contains information about data source.
        input_data_shape: Contains shape of model input layer.

    Returns:
        Resizing factor to scale box coordinates to output frame size.
    """
    frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    _, model_height, model_width, _= input_data_shape
    return max(frame_height, frame_width) / max(model_height, model_width)
