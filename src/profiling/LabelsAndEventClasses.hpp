//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingGuidGenerator.hpp"

#include <armnn/Types.hpp>

namespace armnn
{

namespace profiling
{

class LabelsAndEventClasses
{
public:
    // Labels (string value + GUID)
    static std::string NAME_LABEL;
    static std::string TYPE_LABEL;
    static std::string INDEX_LABEL;
    static std::string BACKENDID_LABEL;
    static ProfilingStaticGuid NAME_GUID;
    static ProfilingStaticGuid TYPE_GUID;
    static ProfilingStaticGuid INDEX_GUID;
    static ProfilingStaticGuid BACKENDID_GUID;

    // Common types
    static std::string LAYER;
    static std::string WORKLOAD;
    static std::string NETWORK;
    static std::string CONNECTION;
    static std::string INFERENCE;
    static std::string WORKLOAD_EXECUTION;
    static ProfilingStaticGuid LAYER_GUID;
    static ProfilingStaticGuid WORKLOAD_GUID;
    static ProfilingStaticGuid NETWORK_GUID;
    static ProfilingStaticGuid CONNECTION_GUID;
    static ProfilingStaticGuid INFERENCE_GUID;
    static ProfilingStaticGuid WORKLOAD_EXECUTION_GUID;

    // Event Class GUIDs
    static ProfilingStaticGuid ARMNN_PROFILING_SOL_EVENT_CLASS;
    static ProfilingStaticGuid ARMNN_PROFILING_EOL_EVENT_CLASS;

private:
    static ProfilingGuidGenerator m_GuidGenerator;
};

} // namespace profiling

} // namespace armnn
