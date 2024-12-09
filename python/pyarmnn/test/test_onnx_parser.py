# Copyright © 2020,2023 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
import os

import pytest
import pyarmnn as ann
import numpy as np


@pytest.fixture()
def parser(shared_data_folder):
    """
    Parse and setup the test network to be used for the tests below
    """

    # create onnx parser
    parser = ann.IOnnxParser()

    # path to model
    path_to_model = os.path.join(shared_data_folder, 'mock_model.onnx')

    # parse onnx binary & create network
    parser.CreateNetworkFromBinaryFile(path_to_model)

    yield parser


def test_onnx_parser_swig_destroy():
    assert ann.IOnnxParser.__swig_destroy__, "There is a swig python destructor defined"
    assert ann.IOnnxParser.__swig_destroy__.__name__ == "delete_IOnnxParser"


def test_check_onnx_parser_swig_ownership(parser):
    # Check to see that SWIG has ownership for parser. This instructs SWIG to take
    # ownership of the return value. This allows the value to be automatically
    # garbage-collected when it is no longer in use
    assert parser.thisown


def test_onnx_parser_get_network_input_binding_info(parser):
    input_binding_info = parser.GetNetworkInputBindingInfo("input")

    tensor = input_binding_info[1]
    assert tensor.GetDataType() == 1
    assert tensor.GetNumDimensions() == 4
    assert tensor.GetNumElements() == 784
    assert tensor.GetQuantizationOffset() == 0
    assert tensor.GetQuantizationScale() == 1


def test_onnx_parser_get_network_output_binding_info(parser):
    output_binding_info = parser.GetNetworkOutputBindingInfo("output")

    tensor = output_binding_info[1]
    assert tensor.GetDataType() == 1
    assert tensor.GetNumDimensions() == 4
    assert tensor.GetNumElements() == 10
    assert tensor.GetQuantizationOffset() == 0
    assert tensor.GetQuantizationScale() == 1


def test_onnx_filenotfound_exception(shared_data_folder):
    parser = ann.IOnnxParser()

    # path to model
    path_to_model = os.path.join(shared_data_folder, 'some_unknown_model.onnx')

    # parse onnx binary & create network

    with pytest.raises(RuntimeError) as err:
        parser.CreateNetworkFromBinaryFile(path_to_model)

    # Only check for part of the exception since the exception returns
    # absolute path which will change on different machines.
    assert 'Invalid (null) filename' in str(err.value)


def test_onnx_parser_end_to_end(shared_data_folder):
    parser = ann.IOnnxParser = ann.IOnnxParser()

    network = parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder, 'mock_model.onnx'))

    # load test image data stored in input_onnx.npy
    input_binding_info = parser.GetNetworkInputBindingInfo("input")
    input_tensor_data = np.load(os.path.join(shared_data_folder, 'onnx_parser/input_onnx.npy')).astype(np.float32)

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    preferred_backends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
    opt_network, messages = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    assert 0 == len(messages)

    net_id, messages = runtime.LoadNetwork(opt_network)

    assert "" == messages

    input_tensors = ann.make_input_tensors([input_binding_info], [input_tensor_data])
    output_tensors = ann.make_output_tensors([parser.GetNetworkOutputBindingInfo("output")])

    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

    output = ann.workload_tensors_to_ndarray(output_tensors)

    # Load golden output file for result comparison.
    golden_output = np.load(os.path.join(shared_data_folder, 'onnx_parser/golden_output_onnx.npy'))

    # Check that output matches golden output to 4 decimal places (there are slight rounding differences after this)
    np.testing.assert_almost_equal(output[0], golden_output, decimal=4)
