# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
import os

import pytest

import pyarmnn as ann


class MockIProfiler:
    def __init__(self, json_string):
        self._profile_json = json_string

    def as_json(self):
        return self._profile_json


@pytest.fixture()
def mock_profiler(shared_data_folder):
    path_to_file = os.path.join(shared_data_folder, 'mock_profile_out.json')
    with open(path_to_file, 'r') as file:
        profiler_output = file.read()
        return MockIProfiler(profiler_output)


def test_inference_exec(mock_profiler):
    profiling_data_obj = ann.get_profiling_data(mock_profiler)

    assert (len(profiling_data_obj.inference_data) > 0)
    assert (len(profiling_data_obj.per_workload_execution_data) > 0)

    # Check each total execution time
    assert (profiling_data_obj.inference_data["execution_time"] == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    assert (profiling_data_obj.inference_data["time_unit"] == "us")


@pytest.mark.parametrize("exec_times, unit, backend, workload", [([2, 2,
                                                                   2, 2,
                                                                   2, 2],
                                                                  'us',
                                                                  'CpuRef',
                                                                  'RefSomeMock1dWorkload_Execute_#5'),
                                                                 ([2, 2,
                                                                   2, 2,
                                                                   2, 2],
                                                                  'us',
                                                                  'CpuAcc',
                                                                  'NeonSomeMock2Workload_Execute_#6'),
                                                                 ([2, 2,
                                                                   2, 2,
                                                                   2, 2],
                                                                  'us',
                                                                  'GpuAcc',
                                                                  'ClSomeMock3dWorkload_Execute_#7'),
                                                                 ([2, 2,
                                                                   2, 2,
                                                                   2, 2],
                                                                  'us',
                                                                  'EthosNAcc',
                                                                  'EthosNSomeMock4dWorkload_Execute_#8')
                                                                 ])
def test_profiler_workloads(mock_profiler, exec_times, unit, backend, workload):
    profiling_data_obj = ann.get_profiling_data(mock_profiler)

    work_load_exec = profiling_data_obj.per_workload_execution_data[workload]
    assert work_load_exec["execution_time"] == exec_times
    assert work_load_exec["time_unit"] == unit
    assert work_load_exec["backend"] == backend