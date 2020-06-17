# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""
This file contains shared functions used in the object detection scripts for
preprocessing data, preparing the network and postprocessing.
"""

import os
import cv2
import numpy as np
import pyarmnn as ann


def create_video_writer(video: cv2.VideoCapture, video_path: str, output_path: str):
    """
    Creates a video writer object to write processed frames to file.

    Args:
        video: Video capture object, contains information about data source.
        video_path: User-specified video file path.
        output_path: Optional path to save the processed video.

    Returns:
        Video writer object.
    """
    _, ext = os.path.splitext(video_path)

    if output_path is not None:
        assert os.path.isdir(output_path)

    i, filename = 0, os.path.join(output_path if output_path is not None else str(), f'object_detection_demo{ext}')
    while os.path.exists(filename):
        i += 1
        filename = os.path.join(output_path if output_path is not None else str(), f'object_detection_demo({i}){ext}')

    video_writer = cv2.VideoWriter(filename=filename,
                                   fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps=int(video.get(cv2.CAP_PROP_FPS)),
                                   frameSize=(int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                              int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    return video_writer


def create_network(model_file: str, backends: list):
    """
    Creates a network based on the model file and a list of backends.

    Args:
        model_file: User-specified model file.
        backends: List of backends to optimize network.

    Returns:
        net_id: Unique ID of the network to run.
        runtime: Runtime context for executing inference.
        input_binding_info: Contains essential information about the model input.
        output_binding_info: Used to map output tensor and its memory.
    """
    if not os.path.exists(model_file):
        raise FileNotFoundError(f'Model file not found for: {model_file}')

    # Determine which parser to create based on model file extension
    parser = None
    _, ext = os.path.splitext(model_file)
    if ext == '.tflite':
        parser = ann.ITfLiteParser()
    elif ext == '.pb':
        parser = ann.ITfParser()
    elif ext == '.onnx':
        parser = ann.IOnnxParser()
    assert (parser is not None)
    network = parser.CreateNetworkFromBinaryFile(model_file)

    # Specify backends to optimize network
    preferred_backends = []
    for b in backends:
        preferred_backends.append(ann.BackendId(b))

    # Select appropriate device context and optimize the network for that device
    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)
    opt_network, messages = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(),
                                  ann.OptimizerOptions())
    print(f'Preferred backends: {backends}\n{runtime.GetDeviceSpec()}\n'
          f'Optimization warnings: {messages}')

    # Load the optimized network onto the Runtime device
    net_id, _ = runtime.LoadNetwork(opt_network)

    # Get input and output binding information
    graph_id = parser.GetSubgraphCount() - 1
    input_names = parser.GetSubgraphInputTensorNames(graph_id)
    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
    output_names = parser.GetSubgraphOutputTensorNames(graph_id)
    output_binding_info = []
    for output_name in output_names:
        outBindInfo = parser.GetNetworkOutputBindingInfo(graph_id, output_name)
        output_binding_info.append(outBindInfo)
    return net_id, runtime, input_binding_info, output_binding_info


def dict_labels(labels_file: str):
    """
    Creates a labels dictionary from the input labels file.

    Args:
        labels_file: Default or user-specified file containing the model output labels.

    Returns:
        A dictionary keyed on the classification index with values corresponding to
        labels and randomly generated RGB colors.
    """
    labels_dict = {}
    with open(labels_file, 'r') as labels:
        for index, line in enumerate(labels, 0):
            labels_dict[index] = line.strip('\n'), tuple(np.random.random(size=3) * 255)
        return labels_dict


def resize_with_aspect_ratio(frame: np.ndarray, input_binding_info: tuple):
    """
    Resizes frame while maintaining aspect ratio, padding any empty space.

    Args:
        frame: Captured frame.
        input_binding_info: Contains shape of model input layer.

    Returns:
        Frame resized to the size of model input layer.
    """
    aspect_ratio = frame.shape[1] / frame.shape[0]
    model_height, model_width = list(input_binding_info[1].GetShape())[1:3]

    if aspect_ratio >= 1.0:
        new_height, new_width = int(model_width / aspect_ratio), model_width
        b_padding, r_padding = model_height - new_height, 0
    else:
        new_height, new_width = model_height, int(model_height * aspect_ratio)
        b_padding, r_padding = 0, model_width - new_width

    # Resize and pad any empty space
    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    frame = cv2.copyMakeBorder(frame, top=0, bottom=b_padding, left=0, right=r_padding,
                               borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return frame


def preprocess(frame: np.ndarray, input_binding_info: tuple):
    """
    Takes a frame, resizes, swaps channels and converts data type to match
    model input layer. The converted frame is wrapped in a const tensor
    and bound to the input tensor.

    Args:
        frame: Captured frame from video.
        input_binding_info:  Contains shape and data type of model input layer.

    Returns:
        Input tensor.
    """
    # Swap channels and resize frame to model resolution
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = resize_with_aspect_ratio(frame, input_binding_info)

    # Expand dimensions and convert data type to match model input
    data_type = np.float32 if input_binding_info[1].GetDataType() == ann.DataType_Float32 else np.uint8
    resized_frame = np.expand_dims(np.asarray(resized_frame, dtype=data_type), axis=0)
    assert resized_frame.shape == tuple(input_binding_info[1].GetShape())

    input_tensors = ann.make_input_tensors([input_binding_info], [resized_frame])
    return input_tensors


def execute_network(input_tensors: list, output_tensors: list, runtime, net_id: int) -> np.ndarray:
    """
    Executes inference for the loaded network.

    Args:
        input_tensors: The input frame tensor.
        output_tensors: The output tensor from output node.
        runtime: Runtime context for executing inference.
        net_id: Unique ID of the network to run.

    Returns:
        Inference results as a list of ndarrays.
    """
    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)
    output = ann.workload_tensors_to_ndarray(output_tensors)
    return output


def draw_bounding_boxes(frame: np.ndarray, detections: list, resize_factor, labels: dict):
    """
    Draws bounding boxes around detected objects and adds a label and confidence score.

    Args:
        frame: The original captured frame from video source.
        detections: A list of detected objects in the form [class, [box positions], confidence].
        resize_factor: Resizing factor to scale box coordinates to output frame size.
        labels: Dictionary of labels and colors keyed on the classification index.
    """
    for detection in detections:
        class_idx, box, confidence = [d for d in detection]
        label, color = labels[class_idx][0].capitalize(), labels[class_idx][1]

        # Obtain frame size and resized bounding box positions
        frame_height, frame_width = frame.shape[:2]
        x_min, y_min, x_max, y_max = [int(position * resize_factor) for position in box]

        # Ensure box stays within the frame
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(frame_width, x_max), min(frame_height, y_max)

        # Draw bounding box around detected object
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Create label for detected object class
        label = f'{label} {confidence * 100:.1f}%'
        label_color = (0, 0, 0) if sum(color)>200 else (255, 255, 255)

        # Make sure label always stays on-screen
        x_text, y_text = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)[0][:2]

        lbl_box_xy_min = (x_min, y_min if y_min<25 else y_min - y_text)
        lbl_box_xy_max = (x_min + int(0.55 * x_text), y_min + y_text if y_min<25 else y_min)
        lbl_text_pos = (x_min + 5, y_min + 16 if y_min<25 else y_min - 5)

        # Add label and confidence value
        cv2.rectangle(frame, lbl_box_xy_min, lbl_box_xy_max, color, -1)
        cv2.putText(frame, label, lbl_text_pos, cv2.FONT_HERSHEY_DUPLEX, 0.50,
                    label_color, 1, cv2.LINE_AA)
