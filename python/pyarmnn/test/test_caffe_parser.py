# Copyright © 2020 Arm Ltd. All rights reserved.
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

    # Create caffe parser
    parser = ann.ICaffeParser()

    # Specify path to model
    path_to_model = os.path.join(shared_data_folder, 'mock_model.caffemodel')

    # Specify the tensor shape relative to the input [1, 1, 28, 28]
    tensor_shape = {'Placeholder': ann.TensorShape((1, 1, 28, 28))}

    # Specify the requested_outputs
    requested_outputs = ["output"]

    # Parse caffe binary & create network
    parser.CreateNetworkFromBinaryFile(path_to_model, tensor_shape, requested_outputs)

    yield parser


def test_caffe_parser_swig_destroy():
    assert ann.ICaffeParser.__swig_destroy__, "There is a swig python destructor defined"
    assert ann.ICaffeParser.__swig_destroy__.__name__ == "delete_ICaffeParser"


def test_check_caffe_parser_swig_ownership(parser):
    # Check to see that SWIG has ownership for parser. This instructs SWIG to take
    # ownership of the return value. This allows the value to be automatically
    # garbage-collected when it is no longer in use
    assert parser.thisown


def test_get_network_input_binding_info(parser):
    input_binding_info = parser.GetNetworkInputBindingInfo("Placeholder")

    tensor = input_binding_info[1]
    assert tensor.GetDataType() == 1
    assert tensor.GetNumDimensions() == 4
    assert tensor.GetNumElements() == 784


def test_get_network_output_binding_info(parser):
    output_binding_info1 = parser.GetNetworkOutputBindingInfo("output")

    # Check the tensor info retrieved from GetNetworkOutputBindingInfo
    tensor1 = output_binding_info1[1]

    assert tensor1.GetDataType() == 1
    assert tensor1.GetNumDimensions() == 2
    assert tensor1.GetNumElements() == 10


def test_filenotfound_exception(shared_data_folder):
    parser = ann.ICaffeParser()

    # path to model
    path_to_model = os.path.join(shared_data_folder, 'some_unknown_network.caffemodel')

    # generic tensor shape [1, 1, 1, 1]
    tensor_shape = {'data': ann.TensorShape((1, 1, 1, 1))}

    # requested_outputs
    requested_outputs = [""]

    with pytest.raises(RuntimeError) as err:
        parser.CreateNetworkFromBinaryFile(path_to_model, tensor_shape, requested_outputs)

    # Only check for part of the exception since the exception returns
    # absolute path which will change on different machines.
    assert 'Failed to open graph file' in str(err.value)


def test_caffe_parser_end_to_end(shared_data_folder):
    parser = ann.ICaffeParser = ann.ICaffeParser()

    # Load the network specifying the inputs and outputs
    input_name = "Placeholder"
    tensor_shape = {input_name: ann.TensorShape((1, 1, 28, 28))}
    requested_outputs = ["output"]

    network = parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder, 'mock_model.caffemodel'),
                                                 tensor_shape, requested_outputs)

    # Specify preferred backend
    preferred_backends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]

    input_binding_info = parser.GetNetworkInputBindingInfo(input_name)

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    opt_network, messages = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    assert 0 == len(messages)

    net_id, messages = runtime.LoadNetwork(opt_network)

    assert "" == messages

    # Load test image data stored in input_caffe.npy
    input_tensor_data = np.load(os.path.join(shared_data_folder, 'caffe_parser/input_caffe.npy')).astype(np.float32)
    input_tensors = ann.make_input_tensors([input_binding_info], [input_tensor_data])

    # Load output binding info and
    outputs_binding_info = []
    for output_name in requested_outputs:
        outputs_binding_info.append(parser.GetNetworkOutputBindingInfo(output_name))
    output_tensors = ann.make_output_tensors(outputs_binding_info)

    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

    output_vectors = ann.workload_tensors_to_ndarray(output_tensors)

    # Load golden output file for result comparison.
    expected_output = np.load(os.path.join(shared_data_folder, 'caffe_parser/golden_output_caffe.npy'))

    # Check that output matches golden output to 4 decimal places (there are slight rounding differences after this)
    np.testing.assert_almost_equal(output_vectors[0], expected_output, 4)
