# Copyright © 2020-2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""
This file contains helper functions for reading video/image data and
 pre/postprocessing of video/image data using OpenCV.
"""

import os

import cv2
import numpy as np


def preprocess(frame: np.ndarray, input_data_type, input_data_shape: tuple, is_normalised: bool,
               keep_aspect_ratio: bool=True):
    """
    Takes a frame, resizes, swaps channels and converts data type to match
    model input layer.

    Args:
        frame: Captured frame from video.
        input_data_type:  Contains data type of model input layer.
        input_data_shape: Contains shape of model input layer.
        is_normalised: if the input layer expects normalised data
        keep_aspect_ratio: Network executor's input data aspect ratio

    Returns:
        Input tensor.
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if keep_aspect_ratio:
        # Swap channels and resize frame to model resolution
        resized_frame = resize_with_aspect_ratio(frame, input_data_shape)
    else:
        # select the height and width from input_data_shape
        frame_height = input_data_shape[1]
        frame_width = input_data_shape[2]
        resized_frame = cv2.resize(frame, (frame_width, frame_height))
    # Expand dimensions and convert data type to match model input
    if np.float32 == input_data_type:
        data_type = np.float32
        if is_normalised:
            resized_frame = resized_frame.astype("float32")/255
    else:
        data_type = np.uint8

    resized_frame = np.expand_dims(np.asarray(resized_frame, dtype=data_type), axis=0)
    assert resized_frame.shape == input_data_shape

    return resized_frame


def resize_with_aspect_ratio(frame: np.ndarray, input_data_shape: tuple):
    """
    Resizes frame while maintaining aspect ratio, padding any empty space.

    Args:
        frame: Captured frame.
        input_data_shape: Contains shape of model input layer.

    Returns:
        Frame resized to the size of model input layer.
    """
    aspect_ratio = frame.shape[1] / frame.shape[0]
    _, model_height, model_width, _ = input_data_shape

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
                                   fourcc=get_source_encoding_int(video),
                                   fps=int(video.get(cv2.CAP_PROP_FPS)),
                                   frameSize=(int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                              int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    return video_writer


def init_video_file_capture(video_path: str, output_path: str):
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


def init_video_stream_capture(video_source: int):
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
        label_color = (0, 0, 0) if sum(color) > 200 else (255, 255, 255)

        # Make sure label always stays on-screen
        x_text, y_text = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)[0][:2]

        lbl_box_xy_min = (x_min, y_min if y_min<25 else y_min - y_text)
        lbl_box_xy_max = (x_min + int(0.55 * x_text), y_min + y_text if y_min<25 else y_min)
        lbl_text_pos = (x_min + 5, y_min + 16 if y_min < 25 else y_min - 5)

        # Add label and confidence value
        cv2.rectangle(frame, lbl_box_xy_min, lbl_box_xy_max, color, -1)
        cv2.putText(frame, label, lbl_text_pos, cv2.FONT_HERSHEY_DUPLEX, 0.50,
                    label_color, 1, cv2.LINE_AA)


def get_source_encoding_int(video_capture):
    return int(video_capture.get(cv2.CAP_PROP_FOURCC))


def crop_bounding_box_object(input_frame: np.ndarray, x_min: float, y_min: float, x_max: float, y_max: float):
    """
        Creates a cropped image based on x and y coordinates.

        Args:
            input_frame: Image to crop
            x_min, y_min, x_max, y_max: Coordinates of the bounding box

        Returns:
            Cropped image
    """
    # Adding +1 to exclude the bounding box pixels.
    cropped_image = input_frame[int(y_min) + 1:int(y_max), int(x_min) + 1:int(x_max)]
    return cropped_image
