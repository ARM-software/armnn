//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "IProfilingService.hpp"
#include "ProfilingService.hpp"

namespace arm
{

namespace pipe
{

std::unique_ptr<IProfilingService> IProfilingService::CreateProfilingService(
    uint16_t maxGlobalCounterId,
    IInitialiseProfilingService& initialiser,
    armnn::Optional<IReportStructure&> reportStructure)
{
    return std::make_unique<ProfilingService>(maxGlobalCounterId, initialiser, reportStructure);
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
