//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingGuidGenerator.hpp"

#include "DllExport.hpp"

namespace arm
{

namespace pipe
{

class LabelsAndEventClasses
{
public:
    // Labels (string value + GUID)
    ARMNN_DLLEXPORT static std::string EMPTY_LABEL;
    ARMNN_DLLEXPORT static std::string NAME_LABEL;
    ARMNN_DLLEXPORT static std::string TYPE_LABEL;
    ARMNN_DLLEXPORT static std::string INDEX_LABEL;
    ARMNN_DLLEXPORT static std::string BACKENDID_LABEL;
    ARMNN_DLLEXPORT static std::string CHILD_LABEL;
    ARMNN_DLLEXPORT static std::string EXECUTION_OF_LABEL;
    ARMNN_DLLEXPORT static std::string PROCESS_ID_LABEL;
    ARMNN_DLLEXPORT static ProfilingStaticGuid EMPTY_GUID;
    ARMNN_DLLEXPORT static ProfilingStaticGuid NAME_GUID;
    ARMNN_DLLEXPORT static ProfilingStaticGuid TYPE_GUID;
    ARMNN_DLLEXPORT static ProfilingStaticGuid INDEX_GUID;
    ARMNN_DLLEXPORT static ProfilingStaticGuid BACKENDID_GUID;
    ARMNN_DLLEXPORT static ProfilingStaticGuid CHILD_GUID;
    ARMNN_DLLEXPORT static ProfilingStaticGuid EXECUTION_OF_GUID;
    ARMNN_DLLEXPORT static ProfilingStaticGuid PROCESS_ID_GUID;

    // Common types
    ARMNN_DLLEXPORT static std::string LAYER;
    ARMNN_DLLEXPORT static std::string WORKLOAD;
    ARMNN_DLLEXPORT static std::string NETWORK;
    ARMNN_DLLEXPORT static std::string CONNECTION;
    ARMNN_DLLEXPORT static std::string INFERENCE;
    ARMNN_DLLEXPORT static std::string WORKLOAD_EXECUTION;
    ARMNN_DLLEXPORT static ProfilingStaticGuid LAYER_GUID;
    ARMNN_DLLEXPORT static ProfilingStaticGuid WORKLOAD_GUID;
    ARMNN_DLLEXPORT static ProfilingStaticGuid NETWORK_GUID;
    ARMNN_DLLEXPORT static ProfilingStaticGuid CONNECTION_GUID;
    ARMNN_DLLEXPORT static ProfilingStaticGuid INFERENCE_GUID;
    ARMNN_DLLEXPORT static ProfilingStaticGuid WORKLOAD_EXECUTION_GUID;

    // Event Class GUIDs
    // Start of Life (SOL)
    ARMNN_DLLEXPORT static ProfilingStaticGuid ARMNN_PROFILING_SOL_EVENT_CLASS;
    ARMNN_DLLEXPORT static std::string ARMNN_PROFILING_SOL_EVENT_CLASS_NAME;
    ARMNN_DLLEXPORT static ProfilingStaticGuid ARMNN_PROFILING_SOL_EVENT_CLASS_NAME_GUID;
    // End of Life (EOL)
    ARMNN_DLLEXPORT static ProfilingStaticGuid ARMNN_PROFILING_EOL_EVENT_CLASS;
    ARMNN_DLLEXPORT static std::string ARMNN_PROFILING_EOL_EVENT_CLASS_NAME;
    ARMNN_DLLEXPORT static ProfilingStaticGuid ARMNN_PROFILING_EOL_EVENT_CLASS_NAME_GUID;

private:
    static ProfilingGuidGenerator m_GuidGenerator;
};

} // namespace pipe

} // namespace arm
