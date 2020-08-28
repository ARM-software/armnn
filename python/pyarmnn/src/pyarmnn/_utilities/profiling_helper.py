# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
"""
This file contains functions relating to the use of the Arm NN profiler within PyArmNN.
"""
import json
from collections import namedtuple

ProfilerData = namedtuple('ProfilerData', ['inference_data', 'per_workload_execution_data'])
ProfilerData.__doc__ = """Container to hold the profiling inference data, and the profiling data per workload.

Contains:
    inference_data (dict): holds end-to-end inference performance data. Keys:
                           'time_unit' - timer units.
                           'execution_time' - list of total inference execution times for each inference run.
    per_workload_execution_data (dict): holds per operation performance data, key is a operation name
                                        Each operation has 
                                        'time_unit' - timer units.   
                                        'execution_time' - list of total execution times for each inference run.
                                        'backend' - backend used for this operation.

Examples:

    >>> data = get_profiling_data(profiler)
    >>> print(data)
    >>> ProfilerData(inference_data={'time_unit': 'us',
                                     'execution_time': [8901372.972]},
                    per_workload_execution_data={'CopyMemGeneric_Execute_#3': {'time_unit': 'us',
                                                                               'execution_time': [28.941],
                                                                               'backend': 'Unknown'},
                                                 'RefConvolution2dWorkload_Execute_#5': {'time_unit': 'us',
                                                                                         'execution_time': [126838.071],
                                                                                         'backend': 'CpuRef'},
                                                 'RefDepthwiseConvolution2dWorkload_Execute_#6': {'time_unit': 'us',
                                                                                                  'execution_time': [49886.208],
                                                                                                  'backend': 'CpuRef'}
                                                 ...etc
                                                 }
                    )
"""


def get_profiling_data(profiler: 'IProfiler') -> ProfilerData:
    """Reads IProfiler object passed in, extracts the relevant data
        and returns it in a ProfilerData container.

        Args:
            profiler (IProfiler): The IProfiler object to be parsed.

        Returns:
            ProfilerData: A container containing the relevant data extracted from the Profiler output.
    """

    top_level_dict = json.loads(profiler.as_json())
    armnn_data = top_level_dict["ArmNN"]
    #Get the inference measurements dict, this will be just one value for key starting with "inference_measurements"
    inference_measurements = [v for k, v in armnn_data.items() if k.startswith("inference_measurements_")][0]

    #Get the execution data dict, this will be just one value for key starting with "Execute_"
    execution_data = [v for k, v in inference_measurements.items() if k.startswith("Execute_")][0]

    workload_data = {}
    inference_data = {}
    for exec_key, exec_value in execution_data.items():
        # Check all items with a type.
        if "type" in exec_value and exec_value["type"] == "Event":
            for event_key, event_value in exec_value.items():
                if event_key.startswith("Wall clock time_#") and event_value["type"] == "Measurement":
                    time_data = __get_wall_clock_times__(event_value)
                    time_data["backend"] = __get_backend(exec_key)
                    workload_data[exec_key] = time_data
        # This is the total inference time map
        if exec_key.startswith("Wall clock time_#") and exec_value["type"] == "Measurement":
            time_data = __get_wall_clock_times__(exec_value)
            inference_data.update(time_data)
    return ProfilerData(inference_data=inference_data, per_workload_execution_data=workload_data)


def __get_wall_clock_times__(wall_clock_item):
    execution_times = wall_clock_item["raw"]
    time_data = {}
    raw_data = []
    for time in execution_times:
        raw_data.append(time)
    time_data["time_unit"] = wall_clock_item["unit"]
    time_data["execution_time"] = raw_data
    return time_data


def __get_backend(exec_key):
    if "ref" in exec_key.lower():
        return "CpuRef"
    elif "neon" in exec_key.lower():
        return "CpuAcc"
    elif "cl" in exec_key.lower():
        return "GpuAcc"
    elif "ethos" in exec_key.lower():
        return "EthosNAcc"
    else:
        return "Unknown"