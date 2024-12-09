# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
import os

import pytest
import warnings
import numpy as np

import pyarmnn as ann


@pytest.fixture(scope="function")
def random_runtime(shared_data_folder):
    parser = ann.ITfLiteParser()
    network = parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder, 'mock_model.tflite'))
    preferred_backends = [ann.BackendId('CpuRef')]
    options = ann.CreationOptions()

    runtime = ann.IRuntime(options)

    graphs_count = parser.GetSubgraphCount()

    graph_id = graphs_count - 1
    input_names = parser.GetSubgraphInputTensorNames(graph_id)

    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
    input_tensor_id = input_binding_info[0]

    input_tensor_info = input_binding_info[1]
    input_tensor_info.SetConstant()

    output_names = parser.GetSubgraphOutputTensorNames(graph_id)

    input_data = np.random.randint(255, size=input_tensor_info.GetNumElements(), dtype=np.uint8)

    const_tensor_pair = (input_tensor_id, ann.ConstTensor(input_tensor_info, input_data))

    input_tensors = [const_tensor_pair]

    output_tensors = []

    for index, output_name in enumerate(output_names):
        out_bind_info = parser.GetNetworkOutputBindingInfo(graph_id, output_name)

        out_tensor_info = out_bind_info[1]
        out_tensor_id = out_bind_info[0]

        output_tensors.append((out_tensor_id,
                               ann.Tensor(out_tensor_info)))

    yield preferred_backends, network, runtime, input_tensors, output_tensors


@pytest.fixture(scope='function')
def mock_model_runtime(shared_data_folder):
    parser = ann.ITfLiteParser()
    network = parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder, 'mock_model.tflite'))
    graph_id = 0

    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, "input_1")

    input_tensor_data = np.load(os.path.join(shared_data_folder, 'tflite_parser/input_lite.npy'))

    preferred_backends = [ann.BackendId('CpuRef')]

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    opt_network, messages = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    print(messages)

    net_id, messages = runtime.LoadNetwork(opt_network)

    print(messages)

    input_tensors = ann.make_input_tensors([input_binding_info], [input_tensor_data])

    output_names = parser.GetSubgraphOutputTensorNames(graph_id)
    outputs_binding_info = []

    for output_name in output_names:
        outputs_binding_info.append(parser.GetNetworkOutputBindingInfo(graph_id, output_name))

    output_tensors = ann.make_output_tensors(outputs_binding_info)

    yield runtime, net_id, input_tensors, output_tensors


def test_python_disowns_network(random_runtime):
    preferred_backends = random_runtime[0]
    network = random_runtime[1]
    runtime = random_runtime[2]
    opt_network, _ = ann.Optimize(network, preferred_backends,
                                  runtime.GetDeviceSpec(), ann.OptimizerOptions())

    runtime.LoadNetwork(opt_network)

    assert not opt_network.thisown


def test_load_network(random_runtime):
    preferred_backends = random_runtime[0]
    network = random_runtime[1]
    runtime = random_runtime[2]

    opt_network, _ = ann.Optimize(network, preferred_backends,
                                  runtime.GetDeviceSpec(), ann.OptimizerOptions())

    net_id, messages = runtime.LoadNetwork(opt_network)
    assert "" == messages
    assert net_id == 0


def test_create_runtime_with_external_profiling_enabled():

    options = ann.CreationOptions()

    options.m_ProfilingOptions.m_FileOnly = True
    options.m_ProfilingOptions.m_EnableProfiling = True
    options.m_ProfilingOptions.m_OutgoingCaptureFile = "/tmp/outgoing.txt"
    options.m_ProfilingOptions.m_IncomingCaptureFile = "/tmp/incoming.txt"
    options.m_ProfilingOptions.m_TimelineEnabled = True
    options.m_ProfilingOptions.m_CapturePeriod = 1000
    options.m_ProfilingOptions.m_FileFormat = "JSON"

    runtime = ann.IRuntime(options)

    assert runtime is not None


def test_create_runtime_with_external_profiling_enabled_invalid_options():

    options = ann.CreationOptions()

    options.m_ProfilingOptions.m_FileOnly = True
    options.m_ProfilingOptions.m_EnableProfiling = False
    options.m_ProfilingOptions.m_OutgoingCaptureFile = "/tmp/outgoing.txt"
    options.m_ProfilingOptions.m_IncomingCaptureFile = "/tmp/incoming.txt"
    options.m_ProfilingOptions.m_TimelineEnabled = True
    options.m_ProfilingOptions.m_CapturePeriod = 1000
    options.m_ProfilingOptions.m_FileFormat = "JSON"

    with pytest.raises(RuntimeError) as err:
        runtime = ann.IRuntime(options)

    expected_error_message = "It is not possible to enable timeline reporting without profiling being enabled"
    assert expected_error_message in str(err.value)


def test_load_network_properties_provided(random_runtime):
    preferred_backends = random_runtime[0]
    network = random_runtime[1]
    runtime = random_runtime[2]

    opt_network, _ = ann.Optimize(network, preferred_backends,
                                  runtime.GetDeviceSpec(), ann.OptimizerOptions())

    inputSource = ann.MemorySource_Undefined
    outputSource = ann.MemorySource_Undefined
    properties = ann.INetworkProperties(False, inputSource, outputSource)
    net_id, messages = runtime.LoadNetwork(opt_network, properties)
    assert "" == messages
    assert net_id == 0


def test_network_properties_constructor(random_runtime):
    preferred_backends = random_runtime[0]
    network = random_runtime[1]
    runtime = random_runtime[2]

    opt_network, _ = ann.Optimize(network, preferred_backends,
                                  runtime.GetDeviceSpec(), ann.OptimizerOptions())

    inputSource = ann.MemorySource_Undefined
    outputSource = ann.MemorySource_Undefined
    properties = ann.INetworkProperties(True, inputSource, outputSource)
    assert properties.m_AsyncEnabled == True
    assert properties.m_ProfilingEnabled == False
    assert properties.m_OutputNetworkDetailsMethod == ann.ProfilingDetailsMethod_Undefined
    assert properties.m_InputSource == ann.MemorySource_Undefined
    assert properties.m_OutputSource == ann.MemorySource_Undefined

    net_id, messages = runtime.LoadNetwork(opt_network, properties)
    assert "" == messages
    assert net_id == 0


def test_unload_network_fails_for_invalid_net_id(random_runtime):
    preferred_backends = random_runtime[0]
    network = random_runtime[1]
    runtime = random_runtime[2]

    ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    with pytest.raises(RuntimeError) as err:
        runtime.UnloadNetwork(9)

    expected_error_message = "Failed to unload network."
    assert expected_error_message in str(err.value)


def test_enqueue_workload(random_runtime):
    preferred_backends = random_runtime[0]
    network = random_runtime[1]
    runtime = random_runtime[2]
    input_tensors = random_runtime[3]
    output_tensors = random_runtime[4]

    opt_network, _ = ann.Optimize(network, preferred_backends,
                                  runtime.GetDeviceSpec(), ann.OptimizerOptions())

    net_id, _ = runtime.LoadNetwork(opt_network)
    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)


def test_enqueue_workload_fails_with_empty_input_tensors(random_runtime):
    preferred_backends = random_runtime[0]
    network = random_runtime[1]
    runtime = random_runtime[2]
    input_tensors = []
    output_tensors = random_runtime[4]

    opt_network, _ = ann.Optimize(network, preferred_backends,
                                  runtime.GetDeviceSpec(), ann.OptimizerOptions())

    net_id, _ = runtime.LoadNetwork(opt_network)
    with pytest.raises(RuntimeError) as err:
        runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

    expected_error_message = "Number of inputs provided does not match network."
    assert expected_error_message in str(err.value)


@pytest.mark.x86_64
@pytest.mark.parametrize('count', [5])
def test_multiple_inference_runs_yield_same_result(count, mock_model_runtime):
    """
    Test that results remain consistent among multiple runs of the same inference.
    """
    runtime = mock_model_runtime[0]
    net_id = mock_model_runtime[1]
    input_tensors = mock_model_runtime[2]
    output_tensors = mock_model_runtime[3]

    expected_results = np.array([[4,  85, 108,  29,   8,  16,   0,   2,   5,   0]])

    for _ in range(count):
        runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

        output_vectors = ann.workload_tensors_to_ndarray(output_tensors)

        for i in range(len(expected_results)):
            assert output_vectors[i].all() == expected_results[i].all()


@pytest.mark.aarch64
def test_aarch64_inference_results(mock_model_runtime):

    runtime = mock_model_runtime[0]
    net_id = mock_model_runtime[1]
    input_tensors = mock_model_runtime[2]
    output_tensors = mock_model_runtime[3]

    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

    output_vectors = ann.workload_tensors_to_ndarray(output_tensors)

    expected_outputs = expected_results = np.array([[4,  85, 108,  29,   8,  16,   0,   2,   5,   0]])

    for i in range(len(expected_outputs)):
        assert output_vectors[i].all() == expected_results[i].all()


def test_enqueue_workload_with_profiler(random_runtime):
    """
    Tests ArmNN's profiling extension
    """
    preferred_backends = random_runtime[0]
    network = random_runtime[1]
    runtime = random_runtime[2]
    input_tensors = random_runtime[3]
    output_tensors = random_runtime[4]

    opt_network, _ = ann.Optimize(network, preferred_backends,
                                  runtime.GetDeviceSpec(), ann.OptimizerOptions())
    net_id, _ = runtime.LoadNetwork(opt_network)

    profiler = runtime.GetProfiler(net_id)
    # By default profiling should be turned off:
    assert profiler.IsProfilingEnabled() is False

    # Enable profiling:
    profiler.EnableProfiling(True)
    assert profiler.IsProfilingEnabled() is True

    # Run the inference:
    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

    # Get profile output as a string:
    str_profile = profiler.as_json()

    # Verify that certain markers are present:
    assert len(str_profile) != 0
    assert str_profile.find('\"ArmNN\": {') > 0

    # Get events analysis output as a string:
    str_events_analysis = profiler.event_log()

    assert "Event Sequence - Name | Duration (ms) | Start (ms) | Stop (ms) | Device" in str_events_analysis

    assert profiler.thisown == 0


def test_check_runtime_swig_ownership(random_runtime):
    # Check to see that SWIG has ownership for runtime. This instructs SWIG to take
    # ownership of the return value. This allows the value to be automatically
    # garbage-collected when it is no longer in use
    runtime = random_runtime[2]
    assert runtime.thisown
