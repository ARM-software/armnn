# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
import os

import pytest
import pyarmnn as ann
import numpy as np


@pytest.fixture(scope="function")
def get_tensor_info_input(shared_data_folder):
    """
    Sample input tensor information.
    """
    parser = ann.ITfLiteParser()
    parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder, 'mock_model.tflite'))
    graph_id = 0

    input_binding_info = [parser.GetNetworkInputBindingInfo(graph_id, 'input_1')]

    yield input_binding_info


@pytest.fixture(scope="function")
def get_tensor_info_output(shared_data_folder):
    """
    Sample output tensor information.
    """
    parser = ann.ITfLiteParser()
    parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder, 'mock_model.tflite'))
    graph_id = 0

    output_names = parser.GetSubgraphOutputTensorNames(graph_id)
    outputs_binding_info = []

    for output_name in output_names:
        outputs_binding_info.append(parser.GetNetworkOutputBindingInfo(graph_id, output_name))

    yield outputs_binding_info


def test_make_input_tensors(get_tensor_info_input):
    input_tensor_info = get_tensor_info_input
    input_data = []

    for tensor_id, tensor_info in input_tensor_info:
        input_data.append(np.random.randint(0, 255, size=(1, tensor_info.GetNumElements())).astype(np.uint8))

    input_tensors = ann.make_input_tensors(input_tensor_info, input_data)
    assert len(input_tensors) == 1

    for tensor, tensor_info in zip(input_tensors, input_tensor_info):
        # Because we created ConstTensor function, we cannot check type directly.
        assert type(tensor[1]).__name__ == 'ConstTensor'
        assert str(tensor[1].GetInfo()) == str(tensor_info[1])


def test_make_output_tensors(get_tensor_info_output):
    output_binding_info = get_tensor_info_output

    output_tensors = ann.make_output_tensors(output_binding_info)
    assert len(output_tensors) == 1

    for tensor, tensor_info in zip(output_tensors, output_binding_info):
        assert type(tensor[1]) == ann.Tensor
        assert str(tensor[1].GetInfo()) == str(tensor_info[1])


def test_workload_tensors_to_ndarray(get_tensor_info_output):
    # Check shape and size of output from workload_tensors_to_ndarray matches expected.
    output_binding_info = get_tensor_info_output
    output_tensors = ann.make_output_tensors(output_binding_info)

    data = ann.workload_tensors_to_ndarray(output_tensors)

    for i in range(0, len(output_tensors)):
        assert data[i].shape == tuple(output_tensors[i][1].GetShape())
        assert data[i].size == output_tensors[i][1].GetNumElements()


def test_make_input_tensors_fp16(get_tensor_info_input):
    # Check ConstTensor with float16
    input_tensor_info = get_tensor_info_input
    input_data = []

    for tensor_id, tensor_info in input_tensor_info:
        input_data.append(np.random.randint(0, 255, size=(1, tensor_info.GetNumElements())).astype(np.float16))
        tensor_info.SetDataType(ann.DataType_Float16)  # set datatype to float16

    input_tensors = ann.make_input_tensors(input_tensor_info, input_data)
    assert len(input_tensors) == 1

    for tensor, tensor_info in zip(input_tensors, input_tensor_info):
        # Because we created ConstTensor function, we cannot check type directly.
        assert type(tensor[1]).__name__ == 'ConstTensor'
        assert str(tensor[1].GetInfo()) == str(tensor_info[1])
        assert tensor[1].GetDataType() == ann.DataType_Float16
        assert tensor[1].GetNumElements() == 28*28*1
        assert tensor[1].GetNumBytes() == (28*28*1)*2  # check each element is two byte
