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
from utils import dict_labels, Profiling
from cv_utils import init_video_stream_capture, preprocess, draw_bounding_boxes
import style_transfer


def get_model_processing(model_name: str, video: cv2.VideoCapture, input_data_shape: tuple):
    """
    Gets model-specific information such as model labels and decoding and processing functions.
    The user can include their own network and functions by adding another statement.

    Args:
        model_name: Name of type of supported model.
        video: Video capture object, contains information about data source.
        input_data_shape: Contains shape of model input layer, used for scaling bounding boxes.

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
    video = init_video_stream_capture(args.video_source)
    action_profiler.profiling_stop_and_print_us("Video initialization")
    model_name = args.model_name
    process_output, resize_factor = get_model_processing(args.model_name, video, executor.get_shape())
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

    while True:
        frame_present, frame = video.read()
        frame = cv2.flip(frame, 1)  # Horizontally flip the frame
        if not frame_present:
            raise RuntimeError('Error reading frame from video stream')

        action_profiler.profiling_start()
        if model_name == "ssd_mobilenet_v1":
            input_data = preprocess(frame, executor.get_data_type(), executor.get_shape(), True)
        else:
            input_data = preprocess(frame, executor.get_data_type(), executor.get_shape(), False)

        output_result = executor.run([input_data])
        if not enable_profile:
            print("Running inference...")
        action_profiler.profiling_stop_and_print_us("Running inference...")
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
