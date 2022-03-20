//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingService.hpp"

#include <client/include/IProfilingService.hpp>

namespace arm
{

namespace pipe
{

std::unique_ptr<IProfilingService> IProfilingService::CreateProfilingService(
    uint16_t maxGlobalCounterId,
    IInitialiseProfilingService& initialiser,
    const std::string& softwareInfo,
    const std::string& softwareVersion,
    const std::string& hardwareVersion,
    arm::pipe::Optional<IReportStructure&> reportStructure)
{
    return std::make_unique<ProfilingService>(maxGlobalCounterId,
                                              initialiser,
                                              softwareInfo,
                                              softwareVersion,
                                              hardwareVersion,
                                              reportStructure);
}

ProfilingGuidGenerator IProfilingService::m_GuidGenerator;

ProfilingDynamicGuid IProfilingService::GetNextGuid()
{
    return m_GuidGenerator.NextGuid();
}

ProfilingStaticGuid IProfilingService::GetStaticId(const std::string& str)
{
    return m_GuidGenerator.GenerateStaticId(str);
}

void IProfilingService::ResetGuidGenerator()
{
    m_GuidGenerator.Reset();
}

ProfilingDynamicGuid IProfilingService::NextGuid()
{
    return IProfilingService::GetNextGuid();
}

ProfilingStaticGuid IProfilingService::GenerateStaticId(const std::string& str)
{
    return IProfilingService::GetStaticId(str);
}

} // namespace pipe
} // namespace arm
