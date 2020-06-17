# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Object detection demo that takes a video stream from a device, runs inference
on each frame producing bounding boxes and labels around detected objects,
and displays a window with the latest processed frame.
"""

import os
import cv2
import pyarmnn as ann
from tqdm import tqdm
from argparse import ArgumentParser

from ssd import ssd_processing, ssd_resize_factor
from yolo import yolo_processing, yolo_resize_factor
from utils import create_network, dict_labels, preprocess, execute_network, draw_bounding_boxes


parser = ArgumentParser()
parser.add_argument('--video_source', type=int, default=0,
                    help='Device index to access video stream. Defaults to primary device camera at index 0')
parser.add_argument('--model_file_path', required=True, type=str,
                    help='Path to the Object Detection model to use')
parser.add_argument('--model_name', required=True, type=str,
                    help='The name of the model being used. Accepted options: ssd_mobilenet_v1, yolo_v3_tiny')
parser.add_argument('--label_path', type=str,
                    help='Path to the labelset for the provided model file')
parser.add_argument('--preferred_backends', type=str, nargs='+', default=['CpuAcc', 'CpuRef'],
                    help='Takes the preferred backends in preference order, separated by whitespace, '
                         'for example: CpuAcc GpuAcc CpuRef. Accepted options: [CpuAcc, CpuRef, GpuAcc]. '
                         'Defaults to [CpuAcc, CpuRef]')
args = parser.parse_args()


def init_video(video_source: int):
    """
    Creates a video capture object from a device.

    Args:
        video_source: Device index used to read video stream.

    Returns:
        Video capture object used to capture frames from a video stream.
    """
    video = cv2.VideoCapture(video_source)
    if not video.isOpened:
        raise RuntimeError(f'Failed to open video capture for device with index: {video_source}')
    print('Processing video stream. Press \'Esc\' key to exit the demo.')
    return video


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
        labels = os.path.join('ssd_labels.txt')
        return labels, ssd_processing, ssd_resize_factor(video)
    elif model_name == 'yolo_v3_tiny':
        labels = os.path.join('yolo_labels.txt')
        return labels, yolo_processing, yolo_resize_factor(video, input_binding_info)
    else:
        raise ValueError(f'{model_name} is not a valid model name')


def main(args):
    video = init_video(args.video_source)
    net_id, runtime, input_binding_info, output_binding_info = create_network(args.model_file_path,
                                                                              args.preferred_backends)
    output_tensors = ann.make_output_tensors(output_binding_info)
    labels, process_output, resize_factor = get_model_processing(args.model_name, video, input_binding_info)
    labels = dict_labels(labels if args.label_path is None else args.label_path)

    while True:
        frame_present, frame = video.read()
        frame = cv2.flip(frame, 1)  # Horizontally flip the frame
        if not frame_present:
            raise RuntimeError('Error reading frame from video stream')
        input_tensors = preprocess(frame, input_binding_info)
        inference_output = execute_network(input_tensors, output_tensors, runtime, net_id)
        detections = process_output(inference_output)
        draw_bounding_boxes(frame, detections, resize_factor, labels)
        cv2.imshow('PyArmNN Object Detection Demo', frame)
        if cv2.waitKey(1) == 27:
            print('\nExit key activated. Closing video...')
            break
    video.release(), cv2.destroyAllWindows()


if __name__ == '__main__':
    main(args)
