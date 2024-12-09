# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
import os

import pytest
import pyarmnn as ann
import numpy as np


def test_TfLiteParserOptions_default_values():
    parserOptions = ann.TfLiteParserOptions()
    assert parserOptions.m_InferAndValidate == False
    assert parserOptions.m_StandInLayerForUnsupported == False


@pytest.fixture()
def parser(shared_data_folder):
    """
    Parse and setup the test network to be used for the tests below
    """
    parser = ann.ITfLiteParser()
    parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder, 'mock_model.tflite'))

    yield parser


def test_tflite_parser_swig_destroy():
    assert ann.ITfLiteParser.__swig_destroy__, "There is a swig python destructor defined"
    assert ann.ITfLiteParser.__swig_destroy__.__name__ == "delete_ITfLiteParser"


def test_check_tflite_parser_swig_ownership(parser):
    # Check to see that SWIG has ownership for parser. This instructs SWIG to take
    # ownership of the return value. This allows the value to be automatically
    # garbage-collected when it is no longer in use
    assert parser.thisown


def test_tflite_parser_with_optional_options():
    parserOptions = ann.TfLiteParserOptions()
    parserOptions.m_InferAndValidate = True
    parser = ann.ITfLiteParser(parserOptions)
    assert parser.thisown


def create_with_opt() :
    parserOptions = ann.TfLiteParserOptions()
    parserOptions.m_InferAndValidate = True
    return ann.ITfLiteParser(parserOptions)


def test_tflite_parser_with_optional_options_out_of_scope(shared_data_folder):
    parser = create_with_opt()
    network = parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder, "mock_model.tflite"))

    graphs_count = parser.GetSubgraphCount()
    graph_id = graphs_count - 1

    input_names = parser.GetSubgraphInputTensorNames(graph_id)
    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])

    output_names = parser.GetSubgraphOutputTensorNames(graph_id)

    preferred_backends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    opt_network, messages = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())
    assert 0 == len(messages)

    net_id, messages = runtime.LoadNetwork(opt_network)
    assert "" == messages


def test_tflite_get_sub_graph_count(parser):
    graphs_count = parser.GetSubgraphCount()
    assert graphs_count == 1


def test_tflite_get_network_input_binding_info(parser):
    graphs_count = parser.GetSubgraphCount()
    graph_id = graphs_count - 1

    input_names = parser.GetSubgraphInputTensorNames(graph_id)

    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])

    tensor = input_binding_info[1]
    assert tensor.GetDataType() == 2
    assert tensor.GetNumDimensions() == 4
    assert tensor.GetNumElements() == 784
    assert tensor.GetQuantizationOffset() == 128
    assert tensor.GetQuantizationScale() == 0.007843137718737125


def test_tflite_get_network_output_binding_info(parser):
    graphs_count = parser.GetSubgraphCount()
    graph_id = graphs_count - 1

    output_names = parser.GetSubgraphOutputTensorNames(graph_id)

    output_binding_info1 = parser.GetNetworkOutputBindingInfo(graph_id, output_names[0])

    # Check the tensor info retrieved from GetNetworkOutputBindingInfo
    tensor1 = output_binding_info1[1]

    assert tensor1.GetDataType() == 2
    assert tensor1.GetNumDimensions() == 2
    assert tensor1.GetNumElements() == 10
    assert tensor1.GetQuantizationOffset() == 0
    assert tensor1.GetQuantizationScale() == 0.00390625


def test_tflite_get_subgraph_input_tensor_names(parser):
    graphs_count = parser.GetSubgraphCount()
    graph_id = graphs_count - 1

    input_names = parser.GetSubgraphInputTensorNames(graph_id)

    assert input_names == ('input_1',)


def test_tflite_get_subgraph_output_tensor_names(parser):
    graphs_count = parser.GetSubgraphCount()
    graph_id = graphs_count - 1

    output_names = parser.GetSubgraphOutputTensorNames(graph_id)

    assert output_names[0] == 'dense/Softmax'


def test_tflite_filenotfound_exception(shared_data_folder):
    parser = ann.ITfLiteParser()

    with pytest.raises(RuntimeError) as err:
        parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder, 'some_unknown_network.tflite'))

    # Only check for part of the exception since the exception returns
    # absolute path which will change on different machines.
    assert 'Cannot find the file' in str(err.value)


def test_tflite_parser_end_to_end(shared_data_folder):
    parser = ann.ITfLiteParser()

    network = parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder, "mock_model.tflite"))

    graphs_count = parser.GetSubgraphCount()
    graph_id = graphs_count - 1

    input_names = parser.GetSubgraphInputTensorNames(graph_id)
    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])

    output_names = parser.GetSubgraphOutputTensorNames(graph_id)

    preferred_backends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    opt_network, messages = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())
    assert 0 == len(messages)

    net_id, messages = runtime.LoadNetwork(opt_network)
    assert "" == messages

    # Load test image data stored in input_lite.npy
    input_tensor_data = np.load(os.path.join(shared_data_folder, 'tflite_parser/input_lite.npy'))
    input_tensors = ann.make_input_tensors([input_binding_info], [input_tensor_data])

    output_tensors = []
    for index, output_name in enumerate(output_names):
        out_bind_info = parser.GetNetworkOutputBindingInfo(graph_id, output_name)
        out_tensor_info = out_bind_info[1]
        out_tensor_id = out_bind_info[0]
        output_tensors.append((out_tensor_id,
                               ann.Tensor(out_tensor_info)))

    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

    output_vectors = []
    for index, out_tensor in enumerate(output_tensors):
        output_vectors.append(out_tensor[1].get_memory_area())

    # Load golden output file for result comparison.
    expected_outputs = np.load(os.path.join(shared_data_folder, 'tflite_parser/golden_output_lite.npy'))

    # Check that output matches golden output
    assert (expected_outputs == output_vectors[0]).all()
