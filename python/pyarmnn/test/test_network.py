# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
import os
import stat
import numpy as np

import pytest
import pyarmnn as ann


def test_optimizer_options_default_values():
    opt = ann.OptimizerOptions()
    assert opt.m_ReduceFp32ToFp16 == False
    assert opt.m_Debug == False
    assert opt.m_ReduceFp32ToBf16 == False
    assert opt.m_ImportEnabled == False
    assert opt.m_shapeInferenceMethod == ann.ShapeInferenceMethod_ValidateOnly


def test_optimizer_options_set_values1():
    opt = ann.OptimizerOptions(True, True)
    assert opt.m_ReduceFp32ToFp16 == True
    assert opt.m_Debug == True
    assert opt.m_ReduceFp32ToBf16 == False
    assert opt.m_ImportEnabled == False
    assert opt.m_shapeInferenceMethod == ann.ShapeInferenceMethod_ValidateOnly


def test_optimizer_options_set_values2():
    opt = ann.OptimizerOptions(False, False, True)
    assert opt.m_ReduceFp32ToFp16 == False
    assert opt.m_Debug == False
    assert opt.m_ReduceFp32ToBf16 == True
    assert opt.m_ImportEnabled == False
    assert opt.m_shapeInferenceMethod == ann.ShapeInferenceMethod_ValidateOnly


def test_optimizer_options_set_values3():
    opt = ann.OptimizerOptions(False, False, True, ann.ShapeInferenceMethod_InferAndValidate, True)
    assert opt.m_ReduceFp32ToFp16 == False
    assert opt.m_Debug == False
    assert opt.m_ReduceFp32ToBf16 == True
    assert opt.m_ImportEnabled == True
    assert opt.m_shapeInferenceMethod == ann.ShapeInferenceMethod_InferAndValidate


@pytest.fixture(scope="function")
def get_runtime(shared_data_folder, network_file):
    parser= ann.ITfLiteParser()
    preferred_backends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
    network = parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder, network_file))
    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    yield preferred_backends, network, runtime


@pytest.mark.parametrize("network_file",
                         [
                             'mock_model.tflite',
                         ],
                         ids=['mock_model'])
def test_optimize_executes_successfully(network_file, get_runtime):
    preferred_backends = [ann.BackendId('CpuRef')]
    network = get_runtime[1]
    runtime = get_runtime[2]

    opt_network, messages = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    assert len(messages) == 0, 'With only CpuRef, there should be no warnings irrelevant of architecture.'
    assert opt_network


@pytest.mark.parametrize("network_file",
                         [
                             'mock_model.tflite',
                         ],
                         ids=['mock_model'])
def test_optimize_owned_by_python(network_file, get_runtime):
    preferred_backends = get_runtime[0]
    network = get_runtime[1]
    runtime = get_runtime[2]

    opt_network, _ = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())
    assert opt_network.thisown


@pytest.mark.aarch64
@pytest.mark.parametrize("network_file",
                         [
                             'mock_model.tflite'
                         ],
                         ids=['mock_model'])
def test_optimize_executes_successfully_for_neon_backend_only(network_file, get_runtime):
    preferred_backends = [ann.BackendId('CpuAcc')]
    network = get_runtime[1]
    runtime = get_runtime[2]

    opt_network, messages = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())
    assert 0 == len(messages)
    assert opt_network


@pytest.mark.parametrize("network_file",
                         [
                             'mock_model.tflite'
                         ],
                         ids=['mock_model'])
def test_optimize_fails_for_invalid_backends(network_file, get_runtime):
    invalid_backends = [ann.BackendId('Unknown')]
    network = get_runtime[1]
    runtime = get_runtime[2]

    with pytest.raises(RuntimeError) as err:
        ann.Optimize(network, invalid_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    expected_error_message = "None of the preferred backends [Unknown ] are supported."
    assert expected_error_message in str(err.value)


@pytest.mark.parametrize("network_file",
                         [
                             'mock_model.tflite'
                         ],
                         ids=['mock_model'])
def test_optimize_fails_for_no_backends_specified(network_file, get_runtime):
    empty_backends = []
    network = get_runtime[1]
    runtime = get_runtime[2]

    with pytest.raises(RuntimeError) as err:
        ann.Optimize(network, empty_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    expected_error_message = "Invoked Optimize with no backends specified"
    assert expected_error_message in str(err.value)


@pytest.mark.parametrize("network_file",
                         [
                             'mock_model.tflite'
                         ],
                         ids=['mock_model'])
def test_serialize_to_dot(network_file, get_runtime, tmpdir):
    preferred_backends = get_runtime[0]
    network = get_runtime[1]
    runtime = get_runtime[2]
    opt_network, _ = ann.Optimize(network, preferred_backends,
                                  runtime.GetDeviceSpec(), ann.OptimizerOptions())
    dot_file_path = os.path.join(tmpdir, 'mock_model.dot')
    """Check that serialized file does not exist at the start, gets created after SerializeToDot and is not empty"""
    assert not os.path.exists(dot_file_path)
    opt_network.SerializeToDot(dot_file_path)

    assert os.path.exists(dot_file_path)

    with open(dot_file_path) as res_file:
        expected_data = res_file.read()
        assert len(expected_data) > 1
        assert '[label=< [1,28,28,1] >]' in expected_data


@pytest.mark.x86_64
@pytest.mark.parametrize("network_file",
                         [
                             'mock_model.tflite'
                         ],
                         ids=['mock_model'])
def test_serialize_to_dot_mode_readonly(network_file, get_runtime, tmpdir):
    preferred_backends = get_runtime[0]
    network = get_runtime[1]
    runtime = get_runtime[2]
    opt_network, _ = ann.Optimize(network, preferred_backends,
                                  runtime.GetDeviceSpec(), ann.OptimizerOptions())
    """Create file, write to it and change mode to read-only"""
    dot_file_path = os.path.join(tmpdir, 'mock_model.dot')
    f = open(dot_file_path, "w+")
    f.write("test")
    f.close()
    os.chmod(dot_file_path, stat.S_IREAD)
    assert os.path.exists(dot_file_path)

    with pytest.raises(RuntimeError) as err:
        opt_network.SerializeToDot(dot_file_path)

    expected_error_message = "Failed to open dot file"
    assert expected_error_message in str(err.value)


@pytest.mark.parametrize("method", [
    'AddActivationLayer',
    'AddAdditionLayer',
    'AddArgMinMaxLayer',
    'AddBatchNormalizationLayer',
    'AddBatchToSpaceNdLayer',
    'AddCastLayer',
    'AddChannelShuffleLayer',
    'AddComparisonLayer',
    'AddConcatLayer',
    'AddConstantLayer',
    'AddConvolution2dLayer',
    'AddConvolution3dLayer',
    'AddDepthToSpaceLayer',
    'AddDepthwiseConvolution2dLayer',
    'AddDequantizeLayer',
    'AddDetectionPostProcessLayer',
    'AddDivisionLayer',
    'AddElementwiseUnaryLayer',
    'AddFloorLayer',
    'AddFillLayer',
    'AddFullyConnectedLayer',
    'AddGatherLayer',
    'AddInputLayer',
    'AddInstanceNormalizationLayer',
    'AddLogicalBinaryLayer',
    'AddLogSoftmaxLayer',
    'AddL2NormalizationLayer',
    'AddLstmLayer',
    'AddMaximumLayer',
    'AddMeanLayer',
    'AddMergeLayer',
    'AddMinimumLayer',
    'AddMultiplicationLayer',
    'AddNormalizationLayer',
    'AddOutputLayer',
    'AddPadLayer',
    'AddPermuteLayer',
    'AddPooling2dLayer',
    'AddPreluLayer',
    'AddQuantizeLayer',
    'AddQuantizedLstmLayer',
    'AddRankLayer',
    'AddReduceLayer',
    'AddReshapeLayer',
    'AddResizeLayer',
    'AddShapeLayer',
    'AddSliceLayer',
    'AddSoftmaxLayer',
    'AddSpaceToBatchNdLayer',
    'AddSpaceToDepthLayer',
    'AddSplitterLayer',
    'AddStackLayer',
    'AddStandInLayer',
    'AddStridedSliceLayer',
    'AddSubtractionLayer',
    'AddSwitchLayer',
    'AddTransposeConvolution2dLayer',
    'AddTransposeLayer'
])
def test_network_method_exists(method):
    assert getattr(ann.INetwork, method, None)


def test_fullyconnected_layer_optional_none():
    net = ann.INetwork()
    layer = net.AddFullyConnectedLayer(ann.FullyConnectedDescriptor(),
                                       ann.ConstTensor())

    assert layer


def test_fullyconnected_layer_optional_provided():
    net = ann.INetwork()
    layer = net.AddFullyConnectedLayer(ann.FullyConnectedDescriptor(),
                                       ann.ConstTensor(),
                                       ann.ConstTensor())

    assert layer


def test_fullyconnected_layer_all_args():
    net = ann.INetwork()
    layer = net.AddFullyConnectedLayer(ann.FullyConnectedDescriptor(),
                                       ann.ConstTensor(),
                                       ann.ConstTensor(),
                                       'NAME1')

    assert layer
    assert 'NAME1' == layer.GetName()


def test_DepthwiseConvolution2d_layer_optional_none():
    net = ann.INetwork()
    layer = net.AddDepthwiseConvolution2dLayer(convolution2dDescriptor=ann.DepthwiseConvolution2dDescriptor(),
                                               weights=ann.ConstTensor())

    assert layer


def test_DepthwiseConvolution2d_layer_optional_provided():
    net = ann.INetwork()
    layer = net.AddDepthwiseConvolution2dLayer(convolution2dDescriptor=ann.DepthwiseConvolution2dDescriptor(),
                                               weights=ann.ConstTensor(),
                                               biases=ann.ConstTensor())

    assert layer


def test_DepthwiseConvolution2d_layer_all_args():
    net = ann.INetwork()
    layer = net.AddDepthwiseConvolution2dLayer(convolution2dDescriptor=ann.DepthwiseConvolution2dDescriptor(),
                                               weights=ann.ConstTensor(),
                                               biases=ann.ConstTensor(),
                                               name='NAME1')

    assert layer
    assert 'NAME1' == layer.GetName()


def test_Convolution2d_layer_optional_none():
    net = ann.INetwork()
    layer = net.AddConvolution2dLayer(convolution2dDescriptor=ann.Convolution2dDescriptor(),
                                      weights=ann.ConstTensor())

    assert layer


def test_Convolution2d_layer_optional_provided():
    net = ann.INetwork()
    layer = net.AddConvolution2dLayer(convolution2dDescriptor=ann.Convolution2dDescriptor(),
                                      weights=ann.ConstTensor(),
                                      biases=ann.ConstTensor())

    assert layer


def test_Convolution2d_layer_all_args():
    net = ann.INetwork()
    layer = net.AddConvolution2dLayer(convolution2dDescriptor=ann.Convolution2dDescriptor(),
                                      weights=ann.ConstTensor(),
                                      biases=ann.ConstTensor(),
                                      name='NAME1')

    assert layer
    assert 'NAME1' == layer.GetName()


def test_add_constant_layer_to_fully_connected():

    inputWidth = 1
    inputHeight = 1
    inputChannels = 5
    inputNum = 2

    outputChannels = 3
    outputNum = 2

    inputShape   = ( inputNum, inputChannels, inputHeight, inputWidth )
    outputShape  = ( outputNum, outputChannels )
    weightsShape = ( inputChannels, outputChannels )
    biasShape    = ( outputChannels, )

    input = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [5.0, 4.0, 3.0, 2.0, 1.0]
    ], dtype=np.float32)

    weights = np.array([
        [.5, 2., .5],
        [.5, 2., 1.],
        [.5, 2., 2.],
        [.5, 2., 3.],
        [.5, 2., 4.]
    ], dtype=np.float32)

    biasValues = np.array([10, 20, 30], dtype=np.float32)

    expectedOutput = np.array([
        [0.5 + 1.0 + 1.5 + 2.0 + 2.5 + biasValues[0],
         2.0 + 4.0 + 6.0 + 8.0 + 10. + biasValues[1],
         0.5 + 2.0 + 6.0 + 12. + 20. + biasValues[2]],
        [2.5 + 2.0 + 1.5 + 1.0 + 0.5 + biasValues[0],
         10.0 + 8.0 + 6.0 + 4.0 + 2. + biasValues[1],
         2.5 + 4.0 + 6.0 + 6. + 4.   + biasValues[2]]
    ], dtype=np.float32)

    network = ann.INetwork()

    input_info = ann.TensorInfo(ann.TensorShape(inputShape), ann.DataType_Float32, 0, 0, True)
    input_tensor = ann.ConstTensor(input_info, input)
    input_layer = network.AddInputLayer(0, "input")

    w_info = ann.TensorInfo(ann.TensorShape(weightsShape), ann.DataType_Float32, 0, 0, True)
    w_tensor = ann.ConstTensor(w_info, weights)
    w_layer = network.AddConstantLayer(w_tensor, "weights")

    b_info = ann.TensorInfo(ann.TensorShape(biasShape), ann.DataType_Float32, 0, 0, True)
    b_tensor = ann.ConstTensor(b_info, biasValues)
    b_layer = network.AddConstantLayer(b_tensor, "bias")

    fc_descriptor = ann.FullyConnectedDescriptor()
    fc_descriptor.m_BiasEnabled = True
    fc_descriptor.m_ConstantWeights = True
    fully_connected = network.AddFullyConnectedLayer(fc_descriptor, "fc")

    output_info = ann.TensorInfo(ann.TensorShape(outputShape), ann.DataType_Float32)
    output_tensor = ann.Tensor(output_info, np.zeros([1, 1], dtype=np.float32))
    output = network.AddOutputLayer(0, "output")

    input_layer.GetOutputSlot(0).Connect(fully_connected.GetInputSlot(0))
    w_layer.GetOutputSlot(0).Connect(fully_connected.GetInputSlot(1))
    b_layer.GetOutputSlot(0).Connect(fully_connected.GetInputSlot(2))
    fully_connected.GetOutputSlot(0).Connect(output.GetInputSlot(0))

    input_layer.GetOutputSlot(0).SetTensorInfo(input_info)
    w_layer.GetOutputSlot(0).SetTensorInfo(w_info)
    b_layer.GetOutputSlot(0).SetTensorInfo(b_info)
    fully_connected.GetOutputSlot(0).SetTensorInfo(output_info)

    preferred_backends = [ann.BackendId('CpuRef')]
    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)
    opt_network, messages = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())
    net_id, messages = runtime.LoadNetwork(opt_network)

    input_tensors = [(0, input_tensor)]
    output_tensors = [(0, output_tensor)]
    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

    output_vectors = ann.workload_tensors_to_ndarray(output_tensors)

    assert (output_vectors==expectedOutput).all()
