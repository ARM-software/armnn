//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BackendProfiling.hpp"
#include "RegisterBackendCounters.hpp"

namespace armnn
{

namespace profiling
{

std::unique_ptr<IRegisterBackendCounters>
    BackendProfiling::GetCounterRegistrationInterface(uint16_t currentMaxGlobalCounterID)
{
    return std::make_unique<RegisterBackendCounters>(RegisterBackendCounters(currentMaxGlobalCounterID, m_backendId));
}

std::unique_ptr<ISendTimelinePacket> BackendProfiling::GetSendTimelinePacket()
{
    return m_ProfilingService.GetSendTimelinePacket();
}

IProfilingGuidGenerator& BackendProfiling::GetProfilingGuidGenerator()
{
    // The profiling service is our Guid Generator.
    return m_ProfilingService;
}

CounterStatus BackendProfiling::GetCounterStatus(uint16_t)
{
    return CounterStatus();
}

std::vector<CounterStatus> BackendProfiling::GetActiveCounters()
{
    return std::vector<CounterStatus>();
}

bool BackendProfiling::IsProfilingEnabled() const
{
    return m_ProfilingService.IsProfilingEnabled();
}

}    // namespace profiling
}    // namespace armnn
