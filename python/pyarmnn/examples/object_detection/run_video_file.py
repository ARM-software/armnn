# Copyright © 2020-2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Object detection demo that takes a video file, runs inference on each frame producing
bounding boxes and labels around detected objects, and saves the processed video.
"""

import os
import sys

script_dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(script_dir, '..', 'common'))

import cv2
from tqdm import tqdm
from argparse import ArgumentParser

from ssd import ssd_processing, ssd_resize_factor
from yolo import yolo_processing, yolo_resize_factor
from utils import dict_labels, Profiling
from cv_utils import init_video_file_capture, preprocess, draw_bounding_boxes
import style_transfer


def get_model_processing(model_name: str, video: cv2.VideoCapture, input_data_shape: tuple):
    """
    Gets model-specific information such as model labels and decoding and processing functions.
    The user can include their own network and functions by adding another statement.

    Args:
        model_name: Name of type of supported model.
        video: Video capture object, contains information about data source.
       input_data_shape: Contains shape of model input layer.

    Returns:
        Model labels, decoding and processing functions.
    """
    if model_name == 'ssd_mobilenet_v1':
        return ssd_processing, ssd_resize_factor(video)
    elif model_name == 'yolo_v3_tiny':
        return yolo_processing, yolo_resize_factor(video, input_data_shape)
    else:
        raise ValueError(f'{model_name} is not a valid model name')


def main(args):
    enable_profile = args.profiling_enabled == "true"
    action_profiler = Profiling(enable_profile)
    overall_profiler = Profiling(enable_profile)
    overall_profiler.profiling_start()
    action_profiler.profiling_start()

    if args.tflite_delegate_path is not None:
        from network_executor_tflite import TFLiteNetworkExecutor as NetworkExecutor
        exec_input_args = (args.model_file_path, args.preferred_backends, args.tflite_delegate_path)
    else:
        from network_executor import ArmnnNetworkExecutor as NetworkExecutor
        exec_input_args = (args.model_file_path, args.preferred_backends)

    executor = NetworkExecutor(*exec_input_args)
    action_profiler.profiling_stop_and_print_us("Executor initialization")

    action_profiler.profiling_start()
    video, video_writer, frame_count = init_video_file_capture(args.video_file_path, args.output_video_file_path)
    process_output, resize_factor = get_model_processing(args.model_name, video, executor.get_shape())
    action_profiler.profiling_stop_and_print_us("Video initialization")

    labels = dict_labels(args.label_path, include_rgb=True)

    if all(element is not None for element in [args.style_predict_model_file_path,
                                               args.style_transfer_model_file_path,
                                               args.style_image_path, args.style_transfer_class]):
        style_image = cv2.imread(args.style_image_path)
        action_profiler.profiling_start()
        style_transfer_executor = style_transfer.StyleTransfer(args.style_predict_model_file_path,
                                                               args.style_transfer_model_file_path,
                                                               style_image, args.preferred_backends,
                                                               args.tflite_delegate_path)
        action_profiler.profiling_stop_and_print_us("Style Transfer Executor initialization")

    for _ in tqdm(frame_count, desc='Processing frames'):
        frame_present, frame = video.read()
        if not frame_present:
            continue
        model_name = args.model_name
        if model_name == "ssd_mobilenet_v1":
            input_data = preprocess(frame, executor.get_data_type(), executor.get_shape(), True)
        else:
            input_data = preprocess(frame, executor.get_data_type(), executor.get_shape(), False)

        action_profiler.profiling_start()
        output_result = executor.run([input_data])
        action_profiler.profiling_stop_and_print_us("Running inference")

        detections = process_output(output_result)

        if all(element is not None for element in [args.style_predict_model_file_path,
                                                   args.style_transfer_model_file_path,
                                                   args.style_image_path, args.style_transfer_class]):
            action_profiler.profiling_start()
            frame = style_transfer.create_stylized_detection(style_transfer_executor, args.style_transfer_class,
                                                             frame, detections, resize_factor, labels)
            action_profiler.profiling_stop_and_print_us("Running Style Transfer")
        else:
            draw_bounding_boxes(frame, detections, resize_factor, labels)

        video_writer.write(frame)
    print('Finished processing frames')
    overall_profiler.profiling_stop_and_print_us("Total compute time")
    video.release(), video_writer.release()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video_file_path', required=True, type=str,
                        help='Path to the video file to run object detection on')
    parser.add_argument('--model_file_path', required=True, type=str,
                        help='Path to the Object Detection model to use')
    parser.add_argument('--model_name', required=True, type=str,
                        help='The name of the model being used. Accepted options: ssd_mobilenet_v1, yolo_v3_tiny')
    parser.add_argument('--label_path', required=True, type=str,
                        help='Path to the labelset for the provided model file')
    parser.add_argument('--output_video_file_path', type=str,
                        help='Path to the output video file with detections added in')
    parser.add_argument('--preferred_backends', type=str, nargs='+', default=['CpuAcc', 'CpuRef'],
                        help='Takes the preferred backends in preference order, separated by whitespace, '
                             'for example: CpuAcc GpuAcc CpuRef. Accepted options: [CpuAcc, CpuRef, GpuAcc]. '
                             'Defaults to [CpuAcc, CpuRef]')
    parser.add_argument('--tflite_delegate_path', type=str,
                        help='Enter TensorFlow Lite Delegate file path (.so file). If not entered,'
                             'will use armnn executor')
    parser.add_argument('--profiling_enabled', type=str,
                        help='[OPTIONAL] Enabling this option will print important ML related milestones timing'
                             'information in micro-seconds. By default, this option is disabled.'
                             'Accepted options are true/false.')
    parser.add_argument('--style_predict_model_file_path', type=str,
                        help='Path to the style prediction model to use')
    parser.add_argument('--style_transfer_model_file_path', type=str,
                        help='Path to the style transfer model to use')
    parser.add_argument('--style_image_path', type=str,
                        help='Path to the style image to create stylized frames')
    parser.add_argument('--style_transfer_class', type=str,
                        help='A class to transform its style')

    args = parser.parse_args()
    main(args)
