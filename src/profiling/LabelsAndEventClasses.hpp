//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingGuidGenerator.hpp"

#include <armnn/Types.hpp>
#include <DllExport.hpp>

namespace armnn
{

namespace profiling
{

class LabelsAndEventClasses
{
public:
    // Labels (string value + GUID)
    ARMNN_DLLEXPORT static std::string NAME_LABEL;
    ARMNN_DLLEXPORT static std::string TYPE_LABEL;
    ARMNN_DLLEXPORT static std::string INDEX_LABEL;
    ARMNN_DLLEXPORT static std::string BACKENDID_LABEL;
    ARMNN_DLLEXPORT static ProfilingStaticGuid NAME_GUID;
    ARMNN_DLLEXPORT static ProfilingStaticGuid TYPE_GUID;
    ARMNN_DLLEXPORT static ProfilingStaticGuid INDEX_GUID;
    ARMNN_DLLEXPORT static ProfilingStaticGuid BACKENDID_GUID;

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
    ARMNN_DLLEXPORT static ProfilingStaticGuid ARMNN_PROFILING_SOL_EVENT_CLASS;
    ARMNN_DLLEXPORT static ProfilingStaticGuid ARMNN_PROFILING_EOL_EVENT_CLASS;

private:
    static ProfilingGuidGenerator m_GuidGenerator;
};

} // namespace profiling

} // namespace armnn
