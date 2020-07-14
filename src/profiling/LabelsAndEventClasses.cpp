//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LabelsAndEventClasses.hpp"

namespace armnn
{

namespace profiling
{

ProfilingGuidGenerator LabelsAndEventClasses::m_GuidGenerator;

// Labels (string value + GUID)
std::string LabelsAndEventClasses::EMPTY_LABEL("");
std::string LabelsAndEventClasses::NAME_LABEL("name");
std::string LabelsAndEventClasses::TYPE_LABEL("type");
std::string LabelsAndEventClasses::INDEX_LABEL("index");
std::string LabelsAndEventClasses::BACKENDID_LABEL("backendId");
std::string LabelsAndEventClasses::CHILD_LABEL("child");
std::string LabelsAndEventClasses::EXECUTION_OF_LABEL("execution_of");
std::string LabelsAndEventClasses::PROCESS_ID_LABEL("processId");

ProfilingStaticGuid LabelsAndEventClasses::EMPTY_GUID(0);
ProfilingStaticGuid LabelsAndEventClasses::NAME_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::NAME_LABEL));
ProfilingStaticGuid LabelsAndEventClasses::TYPE_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::TYPE_LABEL));
ProfilingStaticGuid LabelsAndEventClasses::INDEX_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::INDEX_LABEL));
ProfilingStaticGuid LabelsAndEventClasses::BACKENDID_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::BACKENDID_LABEL));
ProfilingStaticGuid LabelsAndEventClasses::CHILD_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::CHILD_LABEL));
ProfilingStaticGuid LabelsAndEventClasses::EXECUTION_OF_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::EXECUTION_OF_LABEL));
ProfilingStaticGuid LabelsAndEventClasses::PROCESS_ID_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::PROCESS_ID_LABEL));

// Common types
std::string LabelsAndEventClasses::LAYER("layer");
std::string LabelsAndEventClasses::WORKLOAD("workload");
std::string LabelsAndEventClasses::NETWORK("network");
std::string LabelsAndEventClasses::CONNECTION("connection");
std::string LabelsAndEventClasses::INFERENCE("inference");
std::string LabelsAndEventClasses::WORKLOAD_EXECUTION("workload_execution");

ProfilingStaticGuid LabelsAndEventClasses::LAYER_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::LAYER));
ProfilingStaticGuid LabelsAndEventClasses::WORKLOAD_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::WORKLOAD));
ProfilingStaticGuid LabelsAndEventClasses::NETWORK_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::NETWORK));
ProfilingStaticGuid LabelsAndEventClasses::CONNECTION_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::CONNECTION));
ProfilingStaticGuid LabelsAndEventClasses::INFERENCE_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::INFERENCE));
ProfilingStaticGuid LabelsAndEventClasses::WORKLOAD_EXECUTION_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::WORKLOAD_EXECUTION));

// Event Class GUIDs
// Start of Life (SOL)
std::string LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS_NAME("start_of_life");
ProfilingStaticGuid LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS_NAME_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS_NAME));
ProfilingStaticGuid LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS(
    m_GuidGenerator.GenerateStaticId("ARMNN_PROFILING_SOL"));
// End of Life (EOL)
std::string LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS_NAME("end_of_life");
ProfilingStaticGuid LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS_NAME_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS_NAME));
ProfilingStaticGuid LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS(
    m_GuidGenerator.GenerateStaticId("ARMNN_PROFILING_EOL"));

} // namespace profiling

} // namespace armnn
