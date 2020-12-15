# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import tflite_runtime.interpreter as tflite
import os
from utils import run_mock_model


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

def test_external_delegate_options_wrong_logging_level(delegate_dir):
    with pytest.raises(ValueError):
        tflite.load_delegate(
            delegate_dir,
            options={"logging-severity": "wrong"})
