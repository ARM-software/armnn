# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
import urllib.request
import cv2
import network_executor_tflite
import cv_utils


def style_transfer_postprocess(preprocessed_frame: np.ndarray, image_shape: tuple):
    """
        Resizes the output frame of style transfer network and changes the color back to original configuration

        Args:
            preprocessed_frame: A preprocessed frame after style transfer.
            image_shape: Contains shape of the original frame before preprocessing.

        Returns:
            Resizing factor to scale coordinates according to image_shape.
    """

    postprocessed_frame = np.squeeze(preprocessed_frame, axis=0)
    # select original height and width from image_shape
    frame_height = image_shape[0]
    frame_width = image_shape[1]
    postprocessed_frame = cv2.resize(postprocessed_frame, (frame_width, frame_height)).astype("float32") * 255
    postprocessed_frame = cv2.cvtColor(postprocessed_frame, cv2.COLOR_RGB2BGR)

    return postprocessed_frame


def create_stylized_detection(style_transfer_executor, style_transfer_class, frame: np.ndarray,
                              detections: list, resize_factor, labels: dict):
    """
        Perform style transfer on a detected class in a frame

        Args:
            style_transfer_executor: The style transfer executor
            style_transfer_class: The class detected to change its style
            frame: The original captured frame from video source.
            detections: A list of detected objects in the form [class, [box positions], confidence].
            resize_factor: Resizing factor to scale box coordinates to output frame size.
            labels: Dictionary of labels and colors keyed on the classification index.
    """
    for detection in detections:
        class_idx, box, confidence = [d for d in detection]
        label = labels[class_idx][0]
        if label.lower() == style_transfer_class.lower():
            # Obtain frame size and resized bounding box positions
            frame_height, frame_width = frame.shape[:2]
            x_min, y_min, x_max, y_max = [int(position * resize_factor) for position in box]

            # Ensure box stays within the frame
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(frame_width, x_max), min(frame_height, y_max)

            # Crop only the detected object
            cropped_frame = cv_utils.crop_bounding_box_object(frame, x_min, y_min, x_max, y_max)

            # Run style_transfer on preprocessed_frame
            stylized_frame = style_transfer_executor.run_style_transfer(cropped_frame)

            # Paste stylized_frame on the original frame in the correct place
            frame[int(y_min)+1:int(y_max),  int(x_min)+1:int(x_max)] = stylized_frame

    return frame


class StyleTransfer:

    def __init__(self, style_predict_model_path: str, style_transfer_model_path: str,
                 style_image: np.ndarray, backends: list, delegate_path: str):
        """
        Creates an inference executor for style predict network, style transfer network,
        list of backends and a style image.

        Args:
            style_predict_model_path: model which is used to create a style bottleneck
            style_transfer_model_path: model which is used to create stylized frames
            style_image: an image to create the style bottleneck
            backends: List of backends to optimize network.
            delegate_path: tflite delegate file path (.so).
        """

        self.style_predict_executor = network_executor_tflite.TFLiteNetworkExecutor(style_predict_model_path, backends,
                                                                                    delegate_path)
        self.style_transfer_executor = network_executor_tflite.TFLiteNetworkExecutor(style_transfer_model_path,
                                                                                     backends,
                                                                                     delegate_path)
        self.style_bottleneck = self.run_style_predict(style_image)

    def get_style_predict_executor_shape(self):
        """
            Get the input shape of the initiated network.

            Returns:
                tuple: The Shape of the network input.
        """
        return self.style_predict_executor.get_shape()

    # Function to run create a style_bottleneck using preprocessed style image.
    def run_style_predict(self, style_image):
        """
            Creates bottleneck tensor for a given style image.

            Args:
                style_image: an image to create the style bottleneck

            Returns:
                style bottleneck tensor
        """
        # The style image has to be preprocessed to (1, 256, 256, 3)
        preprocessed_style_image = cv_utils.preprocess(style_image, self.style_predict_executor.get_data_type(),
                                                       self.style_predict_executor.get_shape(), True, keep_aspect_ratio=False)
        # output[0] is the style bottleneck tensor
        style_bottleneck = self.style_predict_executor.run([preprocessed_style_image])[0]

        return style_bottleneck

    # Run style transform on preprocessed style image
    def run_style_transfer(self, content_image):
        """
            Runs inference for given content_image and style bottleneck to create a stylized image.

            Args:
                content_image:a content image to stylize
        """
        # The content image has to be preprocessed to (1, 384, 384, 3)
        preprocessed_style_image = cv_utils.preprocess(content_image, np.float32,
                                                       self.style_transfer_executor.get_shape(), True, keep_aspect_ratio=False)

        # Transform content image. output[0] is the stylized image
        stylized_image = self.style_transfer_executor.run([preprocessed_style_image, self.style_bottleneck])[0]

        post_stylized_image = style_transfer_postprocess(stylized_image, content_image.shape)

        return post_stylized_image
