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

    # create tf parser
    parser = ann.ITfParser()

    # path to model
    path_to_model = os.path.join(shared_data_folder, 'mock_model.pb')

    # tensor shape [1, 28, 28, 1]
    tensorshape = {'input': ann.TensorShape((1, 28, 28, 1))}

    # requested_outputs
    requested_outputs = ["output"]

    # parse tf binary & create network
    parser.CreateNetworkFromBinaryFile(path_to_model, tensorshape, requested_outputs)

    yield parser


def test_tf_parser_swig_destroy():
    assert ann.ITfParser.__swig_destroy__, "There is a swig python destructor defined"
    assert ann.ITfParser.__swig_destroy__.__name__ == "delete_ITfParser"


def test_check_tf_parser_swig_ownership(parser):
    # Check to see that SWIG has ownership for parser. This instructs SWIG to take
    # ownership of the return value. This allows the value to be automatically
    # garbage-collected when it is no longer in use
    assert parser.thisown


def test_tf_parser_get_network_input_binding_info(parser):
    input_binding_info = parser.GetNetworkInputBindingInfo("input")

    tensor = input_binding_info[1]
    assert tensor.GetDataType() == 1
    assert tensor.GetNumDimensions() == 4
    assert tensor.GetNumElements() == 28*28*1
    assert tensor.GetQuantizationOffset() == 0
    assert tensor.GetQuantizationScale() == 0


def test_tf_parser_get_network_output_binding_info(parser):
    output_binding_info = parser.GetNetworkOutputBindingInfo("output")

    tensor = output_binding_info[1]
    assert tensor.GetDataType() == 1
    assert tensor.GetNumDimensions() == 2
    assert tensor.GetNumElements() == 10
    assert tensor.GetQuantizationOffset() == 0
    assert tensor.GetQuantizationScale() == 0


def test_tf_filenotfound_exception(shared_data_folder):
    parser = ann.ITfParser()

    # path to model
    path_to_model = os.path.join(shared_data_folder, 'some_unknown_model.pb')

    # tensor shape [1, 1, 1, 1]
    tensorshape = {'input': ann.TensorShape((1, 1, 1, 1))}

    # requested_outputs
    requested_outputs = [""]

    # parse tf binary & create network

    with pytest.raises(RuntimeError) as err:
        parser.CreateNetworkFromBinaryFile(path_to_model, tensorshape, requested_outputs)

    # Only check for part of the exception since the exception returns
    # absolute path which will change on different machines.
    assert 'failed to open' in str(err.value)


def test_tf_parser_end_to_end(shared_data_folder):
    parser = ann.ITfParser = ann.ITfParser()

    tensorshape = {'input': ann.TensorShape((1, 28, 28, 1))}
    requested_outputs = ["output"]

    network = parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder, 'mock_model.pb'),
                                                 tensorshape, requested_outputs)

    input_binding_info = parser.GetNetworkInputBindingInfo("input")

    # load test image data stored in input_tf.npy
    input_tensor_data = np.load(os.path.join(shared_data_folder, 'tf_parser/input_tf.npy')).astype(np.float32)

    preferred_backends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    opt_network, messages = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    assert 0 == len(messages)

    net_id, messages = runtime.LoadNetwork(opt_network)

    assert "" == messages

    input_tensors = ann.make_input_tensors([input_binding_info], [input_tensor_data])

    outputs_binding_info = []

    for output_name in requested_outputs:
        outputs_binding_info.append(parser.GetNetworkOutputBindingInfo(output_name))

    output_tensors = ann.make_output_tensors(outputs_binding_info)

    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)
    output_vectors = ann.workload_tensors_to_ndarray(output_tensors)

    # Load golden output file for result comparison.
    golden_output = np.load(os.path.join(shared_data_folder, 'tf_parser/golden_output_tf.npy'))

    # Check that output matches golden output to 4 decimal places (there are slight rounding differences after this)
    np.testing.assert_almost_equal(output_vectors[0], golden_output, decimal=4)
