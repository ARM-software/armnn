//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LabelsAndEventClasses.hpp"

namespace armnn
{

namespace profiling
{

ProfilingGuidGenerator LabelsAndEventClasses::m_GuidGenerator;

// Labels (string value + GUID)
std::string LabelsAndEventClasses::NAME_LABEL("name");
std::string LabelsAndEventClasses::TYPE_LABEL("type");
std::string LabelsAndEventClasses::INDEX_LABEL("index");
std::string LabelsAndEventClasses::BACKENDID_LABEL("backendId");

ProfilingStaticGuid LabelsAndEventClasses::NAME_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::NAME_LABEL));
ProfilingStaticGuid LabelsAndEventClasses::TYPE_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::TYPE_LABEL));
ProfilingStaticGuid LabelsAndEventClasses::INDEX_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::INDEX_LABEL));
ProfilingStaticGuid LabelsAndEventClasses::BACKENDID_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::BACKENDID_LABEL));

// Common types
std::string LabelsAndEventClasses::LAYER("layer");
std::string LabelsAndEventClasses::WORKLOAD("workload");
std::string LabelsAndEventClasses::NETWORK("network");
std::string LabelsAndEventClasses::CONNECTION("connection");

ProfilingStaticGuid LabelsAndEventClasses::LAYER_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::LAYER));
ProfilingStaticGuid LabelsAndEventClasses::WORKLOAD_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::WORKLOAD));
ProfilingStaticGuid LabelsAndEventClasses::NETWORK_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::NETWORK));
ProfilingStaticGuid LabelsAndEventClasses::CONNECTION_GUID(
    m_GuidGenerator.GenerateStaticId(LabelsAndEventClasses::CONNECTION));

// Event Class GUIDs
ProfilingStaticGuid LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS(
    m_GuidGenerator.GenerateStaticId("ARMNN_PROFILING_SOL"));
ProfilingStaticGuid LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS(
    m_GuidGenerator.GenerateStaticId("ARMNN_PROFILING_EOL"));

} // namespace profiling

} // namespace armnn
