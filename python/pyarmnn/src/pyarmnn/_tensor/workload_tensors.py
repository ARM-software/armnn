# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
"""
This file contains functions relating to WorkloadTensors.
WorkloadTensors are the inputTensors and outputTensors that are consumed by IRuntime.EnqueueWorkload.
"""
from typing import Union, List, Tuple
import logging

import numpy as np

from .tensor import Tensor
from .const_tensor import ConstTensor


def make_input_tensors(inputs_binding_info: List[Tuple],
                       input_data: List[np.ndarray]) -> List[Tuple[int, ConstTensor]]:
    """Returns `inputTensors` to be used with `IRuntime.EnqueueWorkload`.

    This is the primary function to call when you want to produce `inputTensors` for `IRuntime.EnqueueWorkload`.
    The output is a list of tuples containing ConstTensors with a corresponding input tensor id.
    The output should be used directly with `IRuntime.EnqueueWorkload`.
    This function works for single or multiple input data and binding information.

    Examples:
        Creating inputTensors.
        >>> import pyarmnn as ann
        >>> import numpy as np
        >>>
        >>> parser = ann.ITfLiteParser()
        >>> ...
        >>> example_image = np.array(...)
        >>> input_binding_info = parser.GetNetworkInputBindingInfo(...)
        >>>
        >>> input_tensors = ann.make_input_tensors([input_binding_info], [example_image])

    Args:
        inputs_binding_info (list of tuples): (int, `TensorInfo`) Binding information for input tensors obtained from
                                              `GetNetworkInputBindingInfo`.
        input_data (list ndarrays): Tensor data to be used for inference.

    Returns:
        list: `inputTensors` - A list of tuples (`int` , `ConstTensor`).


    Raises:
        ValueError: If length of `inputs_binding_info` and `input_data` are not the same.
    """
    if len(inputs_binding_info) != len(input_data):
        raise ValueError("Length of 'inputs_binding_info' does not match length of 'input_data'")

    input_tensors = []

    for in_bind_info, in_data in zip(inputs_binding_info, input_data):
        in_tensor_id = in_bind_info[0]
        in_tensor_info = in_bind_info[1]
        in_tensor_info.SetConstant()
        input_tensors.append((in_tensor_id, ConstTensor(in_tensor_info, in_data)))

    return input_tensors


def make_output_tensors(outputs_binding_info: List[Tuple]) -> List[Tuple[int, Tensor]]:
    """Returns `outputTensors` to be used with `IRuntime.EnqueueWorkload`.

    This is the primary function to call when you want to produce `outputTensors` for `IRuntime.EnqueueWorkload`.
    The output is a list of tuples containing Tensors with a corresponding output tensor id.
    The output should be used directly with `IRuntime.EnqueueWorkload`.

    Examples:
        Creating outputTensors.
        >>> import pyarmnn as ann
        >>>
        >>> parser = ann.ITfLiteParser()
        >>> ...
        >>> output_binding_info = parser.GetNetworkOutputBindingInfo(...)
        >>>
        >>> output_tensors = ann.make_output_tensors([output_binding_info])

    Args:
        outputs_binding_info (list of tuples): (int, `TensorInfo`) Binding information for output tensors obtained from
                                               `GetNetworkOutputBindingInfo`.

    Returns:
        list: `outputTensors` - A list of tuples (`int`, `Tensor`).
    """
    output_tensors = []

    for out_bind_info in outputs_binding_info:
        out_tensor_id = out_bind_info[0]
        out_tensor_info = out_bind_info[1]
        output_tensors.append((out_tensor_id, Tensor(out_tensor_info)))

    return output_tensors


def workload_tensors_to_ndarray(workload_tensors: List[Tuple[int, Union[Tensor, ConstTensor]]]) -> List[np.ndarray]:
    """Returns a list of the underlying tensor data as ndarrays from `inputTensors` or `outputTensors`.

    We refer to `inputTensors` and `outputTensors` as workload tensors because
    they are used with `IRuntime.EnqueueWorkload`.
    Although this function can be used on either `inputTensors` or `outputTensors` the main use of this function
    is to collect results from `outputTensors` after `IRuntime.EnqueueWorkload` has been called.

    Examples:
        Getting results after inference.
        >>> import pyarmnn as ann
        >>>
        >>> ...
        >>> runtime = ann.IRuntime(...)
        >>> ...
        >>> runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)
        >>>
        >>> inference_results = workload_tensors_to_ndarray(output_tensors)

    Args:
        workload_tensors (inputTensors or outputTensors): `inputTensors` or `outputTensors` to get data from. See
                                                          `make_input_tensors` and `make_output_tensors`.

    Returns:
        list: List of `ndarrays` for the underlying tensor data from given `inputTensors` or `outputTensors`.
    """
    arrays = []
    for index, (_, tensor) in enumerate(workload_tensors):
        arrays.append(tensor.get_memory_area().reshape(list(tensor.GetShape())))
        logging.info("Workload tensor {} shape: {}".format(index, tensor.GetShape()))

    return arrays
