# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
"""
This file contains functions relating to the use of the Arm NN profiler within PyArmNN.
"""
import json
from collections import namedtuple
from operator import itemgetter
import itertools

"""Profiling data is in cycles, to get duration in us, divide by clock frequency. Expected clock frequency is 5 MHz."""
ClockFrequencyDivider = 5

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


def get_profiling_data(profiler: 'IProfiler', backends) -> ProfilerData:
    """Reads IProfiler object passed in, extracts the relevant data.
        If EthosNAcc backend is enabled and trace.json profiling file present
        adds EthosN profiling data and returns all profiling data in a ProfilerData container.

        Args:
            profiler (IProfiler): The IProfiler object to be parsed.
            backends: List of preferred backends.

        Returns:
            ProfilerData: A container containing the relevant data extracted from the Profiler output.
    """

    top_level_dict = json.loads(profiler.as_json())
    armnn_data = top_level_dict["ArmNN"]
    inference_measurements = armnn_data["inference_measurements_#1"]
    execution_data = inference_measurements["Execute_#2"]

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
        ethosn_backend = [backend for backend in backends if "EthosNAcc" == str(backend)]
        if ethosn_backend:
            ethosn_profiling_data = get_ethosn_profiling_data()
            if ethosn_profiling_data:
                workload_data.update(ethosn_profiling_data)

    return ProfilerData(inference_data=inference_data, per_workload_execution_data=workload_data)

def get_ethosn_profiling_data(profiling_json_file = 'trace.json'):
    """If profiling is enabled, profiling data will be recorded in the current directory in trace.json file.
    Read the trace.json file to get timings and operation names.

    Args:
        profiling_json_file (str): Name of profiling json file, defaults to trace.json created in current directory.

    Returns:
        dictionary containing EthosN workload_data of the same structure as per_workload_execution_data.
            Each operation has
            'time_unit' - timer units.
            'execution_time' - list of total execution times for each inference run.
            'backend' - backend used for this operation.
"""
    try:
        with open(profiling_json_file, 'r') as trace_file:
            json_objects = json.loads(trace_file.read())

            # Filter python objects with list comprehensions
            per_workload_execution_data = {}
            commands = [command for command in json_objects if command['name'].startswith("COMMAND")]

            mce_ple_commands = [mce_ple_command for mce_ple_command in commands
                                if "OPERATION_MCE_PLE" in mce_ple_command['args']['command_xml'].keys()]
            per_workload_execution_data.update(__get_command_timings_with_op_info__(mce_ple_commands,
                                                                                    "OPERATION_MCE_PLE", "MCE_OP_INFO"))

            ple_only_commands = [mce_ple_command for mce_ple_command in commands
                                if "OPERATION_PLE_ONLY" in mce_ple_command['args']['command_xml'].keys()]
            per_workload_execution_data.update(__get_command_timings_with_op_info__(ple_only_commands,
                                                                                "OPERATION_PLE_ONLY", "PLE_OP_INFO"))

            other_command_names = {"OPERATION_SOFTMAX", "OPERATION_CONVERT", "OPERATION_DUMP_DRAM",
                                   "OPERATION_DUMP_SRAM", "OPERATION_FENCE", "OPERATION_SECTION", "OPERATION_DELAY"}

            for command_name in other_command_names:
                commands_to_parse = [command for command in commands
                                     if command_name in command['args']['command_xml'].keys()]
                per_workload_execution_data.update(__get_command_timings__(commands_to_parse, command_name))

            return per_workload_execution_data
    except FileNotFoundError:
        print("EthosN profiling file not found, not adding profiling data:", profiling_json_file)
        return None
    except Exception as e:
        print("Got exception while trying to parse EthosN profiling data:", e)
        return None


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

def __get_command_timings_with_op_info__(json_objects, operation_name, op_info_name):
    commands_data = {}
    sorted_objects = sorted(json_objects, key=itemgetter('name'))
    for key, group in itertools.groupby(sorted_objects, key=lambda x:x['name']):
        command_objects = list(group)
        time_data = {"time_unit": 'us'}
        raw_data = []
        for command_object in command_objects:
            duration = ( command_object['ts_end'] - command_object['ts_start'] ) / ClockFrequencyDivider
            raw_data.append(duration)
            time_data["execution_time"] = raw_data
        mce_ple_operation_name = command_objects[0]['args']['command_xml'][operation_name][op_info_name]['OPERATION']
        layer_name = "EthosnCommand#" + str(command_objects[0]['args']['command_idx']) + "_" + \
                     mce_ple_operation_name.capitalize()
        time_data["backend"] = __get_backend(layer_name)
        commands_data[layer_name] = time_data
    return commands_data

def __get_command_timings__(json_objects, operation_name):
    commands_data = {}
    sorted_objects = sorted(json_objects, key=itemgetter('name'))
    for key, group in itertools.groupby(sorted_objects, key=lambda x:x['name']):
        command_objects = list(group)
        time_data = {"time_unit": 'us'}
        raw_data = []
        for command_object in command_objects:
            # Profiling data is in cycles, to get duration in us, divide by clock frequency
            duration = ( command_object['ts_end'] - command_object['ts_start'] ) / ClockFrequencyDivider
            raw_data.append(duration)
            time_data["execution_time"] = raw_data
        layer_name = "EthosnCommand#" + str(command_objects[0]['args']['command_idx']) + "_" + \
                     operation_name.capitalize()
        time_data["backend"] = __get_backend(layer_name)
        commands_data[layer_name] = time_data
    return commands_data