# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import os
from typing import List, Tuple

import pyarmnn as ann
import numpy as np

class ArmnnNetworkExecutor:

    def __init__(self, model_file: str, backends: list):
        """
        Creates an inference executor for a given network and a list of backends.

        Args:
            model_file: User-specified model file.
            backends: List of backends to optimize network.
        """
        self.model_file = model_file
        self.backends = backends
        self.network_id, self.runtime, self.input_binding_info, self.output_binding_info = self.create_network()
        self.output_tensors = ann.make_output_tensors(self.output_binding_info)

    def run(self, input_data_list: list) -> List[np.ndarray]:
        """
        Creates input tensors from input data and executes inference with the loaded network.

        Args:
            input_data_list: List of input frames.

        Returns:
            list: Inference results as a list of ndarrays.
        """
        input_tensors = ann.make_input_tensors(self.input_binding_info, input_data_list)
        self.runtime.EnqueueWorkload(self.network_id, input_tensors, self.output_tensors)
        output = ann.workload_tensors_to_ndarray(self.output_tensors)

        return output

    def create_network(self):
        """
        Creates a network based on the model file and a list of backends.

        Returns:
            net_id: Unique ID of the network to run.
            runtime: Runtime context for executing inference.
            input_binding_info: Contains essential information about the model input.
            output_binding_info: Used to map output tensor and its memory.
        """
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f'Model file not found for: {self.model_file}')

        _, ext = os.path.splitext(self.model_file)
        if ext == '.tflite':
            parser = ann.ITfLiteParser()
        else:
            raise ValueError("Supplied model file type is not supported. Supported types are [ tflite ]")

        network = parser.CreateNetworkFromBinaryFile(self.model_file)

        # Specify backends to optimize network
        preferred_backends = []
        for b in self.backends:
            preferred_backends.append(ann.BackendId(b))

        # Select appropriate device context and optimize the network for that device
        options = ann.CreationOptions()
        runtime = ann.IRuntime(options)
        opt_network, messages = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(),
                                             ann.OptimizerOptions())
        print(f'Preferred backends: {self.backends}\n{runtime.GetDeviceSpec()}\n'
              f'Optimization warnings: {messages}')

        # Load the optimized network onto the Runtime device
        net_id, _ = runtime.LoadNetwork(opt_network)

        # Get input and output binding information
        graph_id = parser.GetSubgraphCount() - 1
        input_names = parser.GetSubgraphInputTensorNames(graph_id)
        input_binding_info = []
        for input_name in input_names:
            in_bind_info = parser.GetNetworkInputBindingInfo(graph_id, input_name)
            input_binding_info.append(in_bind_info)
        output_names = parser.GetSubgraphOutputTensorNames(graph_id)
        output_binding_info = []
        for output_name in output_names:
            out_bind_info = parser.GetNetworkOutputBindingInfo(graph_id, output_name)
            output_binding_info.append(out_bind_info)
        return net_id, runtime, input_binding_info, output_binding_info

    def get_data_type(self):
        """
        Get the input data type of the initiated network.

        Returns:
            numpy data type or None if doesn't exist in the if condition.
        """
        if self.input_binding_info[0][1].GetDataType() == ann.DataType_Float32:
            return np.float32
        elif self.input_binding_info[0][1].GetDataType() == ann.DataType_QAsymmU8:
            return np.uint8
        elif self.input_binding_info[0][1].GetDataType() == ann.DataType_QAsymmS8:
            return np.int8
        else:
            return None

    def get_shape(self):
        """
        Get the input shape of the initiated network.

        Returns:
            tuple: The Shape of the network input.
        """
        return tuple(self.input_binding_info[0][1].GetShape())

    def get_input_quantization_scale(self, idx):
        """
        Get the input quantization scale of the initiated network.

        Returns:
            The quantization scale  of the network input.
        """
        return self.input_binding_info[idx][1].GetQuantizationScale()

    def get_input_quantization_offset(self, idx):
        """
        Get the input quantization offset of the initiated network.

        Returns:
            The quantization offset of the network input.
        """
        return self.input_binding_info[idx][1].GetQuantizationOffset()

    def is_output_quantized(self, idx):
        """
        Get True/False if output tensor is quantized or not respectively.

        Returns:
            True if output is quantized and False otherwise.
        """
        return self.output_binding_info[idx][1].IsQuantized()

    def get_output_quantization_scale(self, idx):
        """
        Get the output quantization offset of the initiated network.

        Returns:
            The quantization offset of the network output.
        """
        return self.output_binding_info[idx][1].GetQuantizationScale()

    def get_output_quantization_offset(self, idx):
        """
        Get the output quantization offset of the initiated network.

        Returns:
            The quantization offset of the network output.
        """
        return self.output_binding_info[idx][1].GetQuantizationOffset()

