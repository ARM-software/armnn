# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
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
    parser = ann.IDeserializer()
    parser.CreateNetworkFromBinary(os.path.join(shared_data_folder, 'mock_model.armnn'))

    yield parser


def test_deserializer_swig_destroy():
    assert ann.IDeserializer.__swig_destroy__, "There is a swig python destructor defined"
    assert ann.IDeserializer.__swig_destroy__.__name__ == "delete_IDeserializer"


def test_check_deserializer_swig_ownership(parser):
    # Check to see that SWIG has ownership for parser. This instructs SWIG to take
    # ownership of the return value. This allows the value to be automatically
    # garbage-collected when it is no longer in use
    assert parser.thisown


def test_deserializer_get_network_input_binding_info(parser):
    # use 0 as a dummy value for layer_id, which is unused in the actual implementation
    layer_id = 0
    input_name = 'input_1'

    input_binding_info = parser.GetNetworkInputBindingInfo(layer_id, input_name)

    tensor = input_binding_info[1]
    assert tensor.GetDataType() == 2
    assert tensor.GetNumDimensions() == 4
    assert tensor.GetNumElements() == 784
    assert tensor.GetQuantizationOffset() == 128
    assert tensor.GetQuantizationScale() == 0.007843137718737125


def test_deserializer_get_network_output_binding_info(parser):
    # use 0 as a dummy value for layer_id, which is unused in the actual implementation
    layer_id = 0
    output_name = "dense/Softmax"

    output_binding_info1 = parser.GetNetworkOutputBindingInfo(layer_id, output_name)

    # Check the tensor info retrieved from GetNetworkOutputBindingInfo
    tensor1 = output_binding_info1[1]

    assert tensor1.GetDataType() == 2
    assert tensor1.GetNumDimensions() == 2
    assert tensor1.GetNumElements() == 10
    assert tensor1.GetQuantizationOffset() == 0
    assert tensor1.GetQuantizationScale() == 0.00390625


def test_deserializer_filenotfound_exception(shared_data_folder):
    parser = ann.IDeserializer()

    with pytest.raises(RuntimeError) as err:
        parser.CreateNetworkFromBinary(os.path.join(shared_data_folder, 'some_unknown_network.armnn'))

    # Only check for part of the exception since the exception returns
    # absolute path which will change on different machines.
    assert 'Cannot read the file' in str(err.value)


def test_deserializer_end_to_end(shared_data_folder):
    parser = ann.IDeserializer()

    network = parser.CreateNetworkFromBinary(os.path.join(shared_data_folder, "mock_model.armnn"))

    # use 0 as a dummy value for layer_id, which is unused in the actual implementation
    layer_id = 0
    input_name = 'input_1'
    output_name = 'dense/Softmax'

    input_binding_info = parser.GetNetworkInputBindingInfo(layer_id, input_name)

    preferred_backends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    opt_network, messages = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())
    assert 0 == len(messages)

    net_id, messages = runtime.LoadNetwork(opt_network)
    assert "" == messages

    # Load test image data stored in input_lite.npy
    input_tensor_data = np.load(os.path.join(shared_data_folder, 'deserializer/input_lite.npy'))
    input_tensors = ann.make_input_tensors([input_binding_info], [input_tensor_data])

    output_tensors = []
    out_bind_info = parser.GetNetworkOutputBindingInfo(layer_id, output_name)
    out_tensor_info = out_bind_info[1]
    out_tensor_id = out_bind_info[0]
    output_tensors.append((out_tensor_id,
                           ann.Tensor(out_tensor_info)))

    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

    output_vectors = []
    for index, out_tensor in enumerate(output_tensors):
        output_vectors.append(out_tensor[1].get_memory_area())

    # Load golden output file for result comparison.
    expected_outputs = np.load(os.path.join(shared_data_folder, 'deserializer/golden_output_lite.npy'))

    # Check that output matches golden output
    assert (expected_outputs == output_vectors[0]).all()
