# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import os
from typing import List, Tuple

import numpy as np
from tflite_runtime import interpreter as tflite

class TFLiteNetworkExecutor:

    def __init__(self, model_file: str, backends: list, delegate_path: str):
        """
        Creates an inference executor for a given network and a list of backends.

        Args:
            model_file: User-specified model file.
            backends: List of backends to optimize network.
            delegate_path: tflite delegate file path (.so).
        """
        self.model_file = model_file
        self.backends = backends
        self.delegate_path = delegate_path
        self.interpreter, self.input_details, self.output_details = self.create_network()

    def run(self, input_data_list: list) -> List[np.ndarray]:
        """
        Executes inference for the loaded network.

        Args:
            input_data_list: List of input frames.

        Returns:
            list: Inference results as a list of ndarrays.
        """
        output = []
        for index, input_data in enumerate(input_data_list):
            self.interpreter.set_tensor(self.input_details[index]['index'], input_data)
        self.interpreter.invoke()
        for curr_output in self.output_details:
            output.append(self.interpreter.get_tensor(curr_output['index']))

        return output

    def create_network(self):
        """
        Creates a network based on the model file and a list of backends.

        Returns:
            interpreter: A TensorFlow Lite object for executing inference.
            input_details: Contains essential information about the model input.
            output_details: Used to map output tensor and its memory.
        """

        # Controls whether optimizations are used or not.
        # Please note that optimizations can improve performance in some cases, but it can also
        # degrade the performance in other cases. Accuracy might also be affected.

        optimization_enable = "true"

        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f'Model file not found for: {self.model_file}')

        _, ext = os.path.splitext(self.model_file)
        if ext == '.tflite':
            armnn_delegate = tflite.load_delegate(library=self.delegate_path,
                                                  options={"backends": ','.join(self.backends), "logging-severity": "info",
                                                           "enable-fast-math": optimization_enable,
                                                           "reduce-fp32-to-fp16": optimization_enable})
            interpreter = tflite.Interpreter(model_path=self.model_file,
                                             experimental_delegates=[armnn_delegate])
            interpreter.allocate_tensors()
        else:
            raise ValueError("Supplied model file type is not supported. Supported types are [ tflite ]")

        # Get input and output binding information
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        return interpreter, input_details, output_details

    def get_data_type(self):
        """
        Get the input data type of the initiated network.

        Returns:
            numpy data type or None if doesn't exist in the if condition.
        """
        return self.input_details[0]['dtype']

    def get_shape(self):
        """
        Get the input shape of the initiated network.

        Returns:
            tuple: The Shape of the network input.
        """
        return tuple(self.input_details[0]['shape'])
