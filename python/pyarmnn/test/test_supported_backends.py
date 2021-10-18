# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
import pytest
import pyarmnn as ann


@pytest.fixture()
def get_supported_backends_setup(shared_data_folder):
    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    get_device_spec = runtime.GetDeviceSpec()
    supported_backends = get_device_spec.GetSupportedBackends()

    yield supported_backends


def test_ownership():
    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    device_spec = runtime.GetDeviceSpec()

    assert not device_spec.thisown


def test_to_string():
    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    device_spec = runtime.GetDeviceSpec()
    expected_str = "IDeviceSpec {{ supportedBackends: [" \
                   "{}" \
                   "]}}".format(', '.join(map(lambda b: str(b), device_spec.GetSupportedBackends())))

    assert expected_str == str(device_spec)


def test_get_supported_backends_cpu_ref(get_supported_backends_setup):
    assert "CpuRef" in map(lambda b: str(b), get_supported_backends_setup)


@pytest.mark.aarch64
class TestNoneCpuRefBackends:

    @pytest.mark.parametrize("backend", ["CpuAcc"])
    def test_get_supported_backends_cpu_acc(self, get_supported_backends_setup, backend):
        assert backend in map(lambda b: str(b), get_supported_backends_setup)
