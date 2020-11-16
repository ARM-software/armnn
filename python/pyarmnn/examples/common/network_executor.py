# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import os
from typing import List, Tuple

import pyarmnn as ann
import numpy as np


def create_network(model_file: str, backends: list, input_names: Tuple[str] = (), output_names: Tuple[str] = ()):
    """
    Creates a network based on the model file and a list of backends.

    Args:
        model_file: User-specified model file.
        backends: List of backends to optimize network.
        input_names:
        output_names:

    Returns:
        net_id: Unique ID of the network to run.
        runtime: Runtime context for executing inference.
        input_binding_info: Contains essential information about the model input.
        output_binding_info: Used to map output tensor and its memory.
    """
    if not os.path.exists(model_file):
        raise FileNotFoundError(f'Model file not found for: {model_file}')

    _, ext = os.path.splitext(model_file)
    if ext == '.tflite':
        parser = ann.ITfLiteParser()
    else:
        raise ValueError("Supplied model file type is not supported. Supported types are [ tflite ]")

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
        out_bind_info = parser.GetNetworkOutputBindingInfo(graph_id, output_name)
        output_binding_info.append(out_bind_info)
    return net_id, runtime, input_binding_info, output_binding_info


def execute_network(input_tensors: list, output_tensors: list, runtime, net_id: int) -> List[np.ndarray]:
    """
    Executes inference for the loaded network.

    Args:
        input_tensors: The input frame tensor.
        output_tensors: The output tensor from output node.
        runtime: Runtime context for executing inference.
        net_id: Unique ID of the network to run.

    Returns:
        list: Inference results as a list of ndarrays.
    """
    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)
    output = ann.workload_tensors_to_ndarray(output_tensors)
    return output


class ArmnnNetworkExecutor:

    def __init__(self, model_file: str, backends: list):
        """
        Creates an inference executor for a given network and a list of backends.

        Args:
            model_file: User-specified model file.
            backends: List of backends to optimize network.
        """
        self.network_id, self.runtime, self.input_binding_info, self.output_binding_info = create_network(model_file,
                                                                                                          backends)
        self.output_tensors = ann.make_output_tensors(self.output_binding_info)

    def run(self, input_tensors: list) -> List[np.ndarray]:
        """
        Executes inference for the loaded network.

        Args:
            input_tensors: The input frame tensor.

        Returns:
            list: Inference results as a list of ndarrays.
        """
        return execute_network(input_tensors, self.output_tensors, self.runtime, self.network_id)
