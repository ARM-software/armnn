# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Object detection demo that takes a video file, runs inference on each frame producing
bounding boxes and labels around detected objects, and saves the processed video.
"""

import os
import cv2
import pyarmnn as ann
from tqdm import tqdm
from argparse import ArgumentParser

from ssd import ssd_processing, ssd_resize_factor
from yolo import yolo_processing, yolo_resize_factor
from utils import create_video_writer, create_network, dict_labels, preprocess, execute_network, draw_bounding_boxes


parser = ArgumentParser()
parser.add_argument('--video_file_path', required=True, type=str,
                    help='Path to the video file to run object detection on')
parser.add_argument('--model_file_path', required=True, type=str,
                    help='Path to the Object Detection model to use')
parser.add_argument('--model_name', required=True, type=str,
                    help='The name of the model being used. Accepted options: ssd_mobilenet_v1, yolo_v3_tiny')
parser.add_argument('--label_path', type=str,
                    help='Path to the labelset for the provided model file')
parser.add_argument('--output_video_file_path', type=str,
                    help='Path to the output video file with detections added in')
parser.add_argument('--preferred_backends', type=str, nargs='+', default=['CpuAcc', 'CpuRef'],
                    help='Takes the preferred backends in preference order, separated by whitespace, '
                         'for example: CpuAcc GpuAcc CpuRef. Accepted options: [CpuAcc, CpuRef, GpuAcc]. '
                         'Defaults to [CpuAcc, CpuRef]')
args = parser.parse_args()


def init_video(video_path: str, output_path: str):
    """
    Creates a video capture object from a video file.

    Args:
        video_path: User-specified video file path.
        output_path: Optional path to save the processed video.

    Returns:
        Video capture object to capture frames, video writer object to write processed
        frames to file, plus total frame count of video source to iterate through.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Video file not found for: {video_path}')
    video = cv2.VideoCapture(video_path)
    if not video.isOpened:
        raise RuntimeError(f'Failed to open video capture from file: {video_path}')

    video_writer = create_video_writer(video, video_path, output_path)
    iter_frame_count = range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    return video, video_writer, iter_frame_count


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
    video, video_writer, frame_count = init_video(args.video_file_path, args.output_video_file_path)
    net_id, runtime, input_binding_info, output_binding_info = create_network(args.model_file_path,
                                                                              args.preferred_backends)
    output_tensors = ann.make_output_tensors(output_binding_info)
    labels, process_output, resize_factor = get_model_processing(args.model_name, video, input_binding_info)
    labels = dict_labels(labels if args.label_path is None else args.label_path)

    for _ in tqdm(frame_count, desc='Processing frames'):
        frame_present, frame = video.read()
        if not frame_present:
            continue
        input_tensors = preprocess(frame, input_binding_info)
        inference_output = execute_network(input_tensors, output_tensors, runtime, net_id)
        detections = process_output(inference_output)
        draw_bounding_boxes(frame, detections, resize_factor, labels)
        video_writer.write(frame)
    print('Finished processing frames')
    video.release(), video_writer.release()


if __name__ == '__main__':
    main(args)
