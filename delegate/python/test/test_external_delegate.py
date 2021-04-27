# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import tflite_runtime.interpreter as tflite
import os
from utils import run_mock_model, run_inference, compare_outputs

def test_external_delegate_unknown_options(delegate_dir):
    print(delegate_dir)
    with pytest.raises(ValueError):
        tflite.load_delegate(
            delegate_dir,
            options={"wrong": "wrong"})

def test_external_delegate_options_multiple_backends(delegate_dir):
    tflite.load_delegate(
        delegate_dir,
        options={"backends": "GpuAcc,CpuAcc,CpuRef,Unknown"})


@pytest.mark.GpuAccTest
def test_external_delegate_options_gpu_tuning(delegate_dir, test_data_folder, tmp_path):

    tuning_file = os.path.join(str(tmp_path), "test_gpu.tuning")
    # cleanup previous test run if necessary
    if os.path.exists(tuning_file):
        os.remove(tuning_file)

    # with tuning level 2 a tuning file should be created
    armnn_delegate = tflite.load_delegate(
        delegate_dir,
        options={
            "backends": "GpuAcc",
            "gpu-tuning-level": "2",
            "gpu-tuning-file": tuning_file,
            "logging-severity": "info"})

    run_mock_model(armnn_delegate, test_data_folder)

    # destroy delegate, otherwise tuning file won't be written to file
    armnn_delegate.__del__()
    assert (os.path.exists(tuning_file))

    # if no tuning level is provided it defaults to 0 which means it will use the tuning parameters from a tuning
    # file if one is provided
    armnn_delegate2 = tflite.load_delegate(
        delegate_dir,
        options={
            "backends": "GpuAcc",
            "gpu-tuning-file": tuning_file,
            "logging-severity": "info"})

    run_mock_model(armnn_delegate2, test_data_folder)

    # cleanup
    os.remove(tuning_file)

@pytest.mark.GpuAccTest
def test_external_delegate_options_gpu_cached_network(delegate_dir, test_data_folder, tmp_path):

    binary_file = os.path.join(str(tmp_path), "test_binary.bin")
    # cleanup previous test run if necessary
    if os.path.exists(binary_file):
        os.remove(binary_file)

    # Create blank binary file to write to.
    open(binary_file, 'a').close()
    assert (os.path.exists(binary_file))
    assert (os.stat(binary_file).st_size == 0)

    # Run inference to save cached network.
    armnn_delegate = tflite.load_delegate(
        delegate_dir,
        options={
            "backends": "GpuAcc",
            "save-cached-network": "1",
            "cached-network-filepath": binary_file,
            "logging-severity": "info"})

    run_mock_model(armnn_delegate, test_data_folder)

    # destroy delegate and check if file has been saved.
    armnn_delegate.__del__()
    assert (os.stat(binary_file).st_size != 0)

    # Create second delegate to load in binary file created.
    armnn_delegate2 = tflite.load_delegate(
        delegate_dir,
        options={
            "backends": "GpuAcc",
            "cached-network-filepath": binary_file,
            "logging-severity": "info"})

    run_mock_model(armnn_delegate2, test_data_folder)

    # cleanup
    os.remove(binary_file)

@pytest.mark.GpuAccTest
def test_external_delegate_gpu_fastmath(delegate_dir, test_data_folder):
    # create armnn delegate with enable-fast-math
    # fast-math is only enabled on Conv2d layer, so use conv2d model.
    armnn_delegate = tflite.load_delegate(delegate_dir, options = {'backends': 'GpuAcc',
                                                                   'enable-fast-math': '1',
                                                                   "logging-severity": "info"})

    model_file_name = 'conv2d.tflite'

    inputShape = [ 1, 5, 5, 1 ]
    outputShape = [ 1, 3, 3, 1 ]

    inputValues = [ 1, 5, 2, 3, 5,
                    8, 7, 3, 6, 3,
                    3, 3, 9, 1, 9,
                    4, 1, 8, 1, 3,
                    6, 8, 1, 9, 2 ]

    expectedResult = [ 28, 38, 29,
                       96, 104, 53,
                       31, 55, 24 ]

    input = np.array(inputValues, dtype=np.float32).reshape(inputShape)
    expected_output = np.array(expectedResult, dtype=np.float32).reshape(outputShape)

    # run the inference
    armnn_outputs = run_inference(test_data_folder, model_file_name, [input], [armnn_delegate])

    # check results
    compare_outputs(armnn_outputs, [expected_output])

@pytest.mark.CpuAccTest
def test_external_delegate_cpu_options(capfd, delegate_dir, test_data_folder):
    # create armnn delegate with enable-fast-math and number-of-threads options
    # fast-math is only enabled on Conv2d layer, so use conv2d model.
    armnn_delegate = tflite.load_delegate(delegate_dir, options = {'backends': 'CpuAcc',
                                                                   'enable-fast-math': '1',
                                                                   'number-of-threads': '4',
                                                                   "logging-severity": "info"})

    model_file_name = 'conv2d.tflite'

    inputShape = [ 1, 5, 5, 1 ]
    outputShape = [ 1, 3, 3, 1 ]

    inputValues = [ 1, 5, 2, 3, 5,
                    8, 7, 3, 6, 3,
                    3, 3, 9, 1, 9,
                    4, 1, 8, 1, 3,
                    6, 8, 1, 9, 2 ]

    expectedResult = [ 28, 38, 29,
                       96, 104, 53,
                       31, 55, 24 ]

    input = np.array(inputValues, dtype=np.float32).reshape(inputShape)
    expected_output = np.array(expectedResult, dtype=np.float32).reshape(outputShape)

    # run the inference
    armnn_outputs = run_inference(test_data_folder, model_file_name, [input], [armnn_delegate])

    # check results
    compare_outputs(armnn_outputs, [expected_output])

    captured = capfd.readouterr()
    assert 'Set CPPScheduler to Linear mode, with 4 threads to use' in captured.out

def test_external_delegate_options_wrong_logging_level(delegate_dir):
    with pytest.raises(ValueError):
        tflite.load_delegate(
            delegate_dir,
            options={"logging-severity": "wrong"})

def test_external_delegate_options_debug(capfd, delegate_dir, test_data_folder):
    # create armnn delegate with debug option
    armnn_delegate = tflite.load_delegate(delegate_dir, options = {'backends': 'CpuRef', 'debug-data': '1'})

    model_file_name = 'fp32_model.tflite'

    tensor_shape = [1, 2, 2, 1]

    input0 = np.array([1, 2, 3, 4], dtype=np.float32).reshape(tensor_shape)
    input1 = np.array([2, 2, 3, 4], dtype=np.float32).reshape(tensor_shape)
    inputs = [input0, input0, input1]
    expected_output = np.array([1, 2, 2, 2], dtype=np.float32).reshape(tensor_shape)

    # run the inference
    armnn_outputs = run_inference(test_data_folder, model_file_name, inputs, [armnn_delegate])

    # check results
    compare_outputs(armnn_outputs, [expected_output])

    captured = capfd.readouterr()
    assert 'layerGuid' in captured.out


def test_external_delegate_options_fp32_to_fp16(capfd, delegate_dir, test_data_folder):
    # create armnn delegate with reduce-fp32-to-fp16 option
    armnn_delegate = tflite.load_delegate(delegate_dir, options = {'backends': 'CpuRef',
                                                                   'debug-data': '1',
                                                                   'reduce-fp32-to-fp16': '1'})

    model_file_name = 'fp32_model.tflite'

    tensor_shape = [1, 2, 2, 1]

    input0 = np.array([1, 2, 3, 4], dtype=np.float32).reshape(tensor_shape)
    input1 = np.array([2, 2, 3, 4], dtype=np.float32).reshape(tensor_shape)
    inputs = [input0, input0, input1]
    expected_output = np.array([1, 2, 2, 2], dtype=np.float32).reshape(tensor_shape)

    # run the inference
    armnn_outputs = run_inference(test_data_folder, model_file_name, inputs, [armnn_delegate])

    # check results
    compare_outputs(armnn_outputs, [expected_output])

    captured = capfd.readouterr()
    assert 'convert_fp32_to_fp16' in captured.out
    assert 'convert_fp16_to_fp32' in captured.out

def test_external_delegate_options_fp32_to_bf16(capfd, delegate_dir, test_data_folder):
    # create armnn delegate with reduce-fp32-to-bf16 option
    armnn_delegate = tflite.load_delegate(delegate_dir, options = {'backends': 'CpuRef',
                                                                   'debug-data': '1',
                                                                   'reduce-fp32-to-bf16': '1'})

    model_file_name = 'conv2d.tflite'

    inputShape = [ 1, 5, 5, 1 ]
    outputShape = [ 1, 3, 3, 1 ]

    inputValues = [ 1, 5, 2, 3, 5,
                    8, 7, 3, 6, 3,
                    3, 3, 9, 1, 9,
                    4, 1, 8, 1, 3,
                    6, 8, 1, 9, 2 ]

    expectedResult = [ 28, 38, 29,
                       96, 104, 53,
                       31, 55, 24 ]

    input = np.array(inputValues, dtype=np.float32).reshape(inputShape)
    expected_output = np.array(expectedResult, dtype=np.float32).reshape(outputShape)

    # run the inference
    armnn_outputs = run_inference(test_data_folder, model_file_name, [input], [armnn_delegate])

    # check results
    compare_outputs(armnn_outputs, [expected_output])

    captured = capfd.readouterr()
    assert 'convert_fp32_to_bf16' in captured.out

def test_external_delegate_options_memory_import(delegate_dir, test_data_folder):
    # create armnn delegate with memory-import option
    armnn_delegate = tflite.load_delegate(delegate_dir, options = {'backends': 'CpuAcc,CpuRef',
                                                                   'memory-import': '1'})

    model_file_name = 'fallback_model.tflite'

    tensor_shape = [1, 2, 2, 1]

    input0 = np.array([1, 2, 3, 4], dtype=np.uint8).reshape(tensor_shape)
    input1 = np.array([2, 2, 3, 4], dtype=np.uint8).reshape(tensor_shape)
    inputs = [input0, input0, input1]
    expected_output = np.array([1, 2, 2, 2], dtype=np.uint8).reshape(tensor_shape)

    # run the inference
    armnn_outputs = run_inference(test_data_folder, model_file_name, inputs, [armnn_delegate])

    # check results
    compare_outputs(armnn_outputs, [expected_output])