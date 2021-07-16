# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Object detection demo that takes a video stream from a device, runs inference
on each frame producing bounding boxes and labels around detected objects,
and displays a window with the latest processed frame.
"""

import os
import sys
script_dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(script_dir, '..', 'common'))

import cv2
from argparse import ArgumentParser

from ssd import ssd_processing, ssd_resize_factor
from yolo import yolo_processing, yolo_resize_factor
from utils import dict_labels
from cv_utils import init_video_stream_capture, preprocess, draw_bounding_boxes
from network_executor import ArmnnNetworkExecutor


def get_model_processing(model_name: str, video: cv2.VideoCapture, input_binding_info: tuple):
    """
    Gets model-specific information such as model labels and decoding and processing functions.
    The user can include their own network and functions by adding another statement.

    Args:
        model_name: Name of type of supported model.
        video: Video capture object, contains information about data source.
        input_binding_info: Contains shape of model input layer, used for scaling bounding boxes.

    Returns:
        Model labels, decoding and processing functions.
    """
    if model_name == 'ssd_mobilenet_v1':
        return ssd_processing, ssd_resize_factor(video)
    elif model_name == 'yolo_v3_tiny':
        return yolo_processing, yolo_resize_factor(video, input_binding_info)
    else:
        raise ValueError(f'{model_name} is not a valid model name')


def main(args):
    video = init_video_stream_capture(args.video_source)
    executor = ArmnnNetworkExecutor(args.model_file_path, args.preferred_backends)

    model_name = args.model_name
    process_output, resize_factor = get_model_processing(args.model_name, video, executor.input_binding_info)
    labels = dict_labels(args.label_path, include_rgb=True)

    while True:
        frame_present, frame = video.read()
        frame = cv2.flip(frame, 1)  # Horizontally flip the frame
        if not frame_present:
            raise RuntimeError('Error reading frame from video stream')

        if model_name == "ssd_mobilenet_v1":
            input_tensors = preprocess(frame, executor.input_binding_info, True)
        else:
            input_tensors = preprocess(frame, executor.input_binding_info, False)
        print("Running inference...")
        output_result = executor.run(input_tensors)
        detections = process_output(output_result)
        draw_bounding_boxes(frame, detections, resize_factor, labels)
        cv2.imshow('PyArmNN Object Detection Demo', frame)
        if cv2.waitKey(1) == 27:
            print('\nExit key activated. Closing video...')
            break
    video.release(), cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video_source', type=int, default=0,
                        help='Device index to access video stream. Defaults to primary device camera at index 0')
    parser.add_argument('--model_file_path', required=True, type=str,
                        help='Path to the Object Detection model to use')
    parser.add_argument('--model_name', required=True, type=str,
                        help='The name of the model being used. Accepted options: ssd_mobilenet_v1, yolo_v3_tiny')
    parser.add_argument('--label_path', required=True, type=str,
                        help='Path to the labelset for the provided model file')
    parser.add_argument('--preferred_backends', type=str, nargs='+', default=['CpuAcc', 'CpuRef'],
                        help='Takes the preferred backends in preference order, separated by whitespace, '
                             'for example: CpuAcc GpuAcc CpuRef. Accepted options: [CpuAcc, CpuRef, GpuAcc]. '
                             'Defaults to [CpuAcc, CpuRef]')
    args = parser.parse_args()
    main(args)
