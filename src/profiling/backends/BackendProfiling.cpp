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
    return std::make_unique<RegisterBackendCounters>(
        RegisterBackendCounters(currentMaxGlobalCounterID, m_BackendId, m_ProfilingService));
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

void BackendProfiling::ReportCounters(const std::vector<Timestamp>& timestamps)
{
    for (const auto& timestampInfo : timestamps)
    {
        std::vector<CounterValue> backendCounterValues = timestampInfo.counterValues;
        for_each(backendCounterValues.begin(), backendCounterValues.end(), [&](CounterValue& backendCounterValue)
        {
            // translate the counterId to globalCounterId
            backendCounterValue.counterId = m_ProfilingService.GetCounterMappings().GetGlobalId(
                backendCounterValue.counterId, m_BackendId);
        });

        // Send Periodic Counter Capture Packet for the Timestamp
        m_ProfilingService.GetSendCounterPacket().SendPeriodicCounterCapturePacket(
            timestampInfo.timestamp, backendCounterValues);
    }
}

CounterStatus BackendProfiling::GetCounterStatus(uint16_t backendCounterId)
{
    uint16_t globalCounterId = m_ProfilingService.GetCounterMappings().GetGlobalId(backendCounterId, m_BackendId);
    CaptureData captureData = m_ProfilingService.GetCaptureData();

    CounterStatus counterStatus(backendCounterId, globalCounterId, false, 0);

    if (captureData.IsCounterIdInCaptureData(globalCounterId))
    {
        counterStatus.m_Enabled = true;
        counterStatus.m_SamplingRateInMicroseconds = captureData.GetCapturePeriod();
    }

    return counterStatus;
}

std::vector<CounterStatus> BackendProfiling::GetActiveCounters()
{
    CaptureData captureData = m_ProfilingService.GetCaptureData();

    const std::vector<uint16_t>& globalCounterIds = captureData.GetCounterIds();
    std::vector<CounterStatus> activeCounterIds;

    for (auto globalCounterId : globalCounterIds) {
        // Get pair of local counterId and backendId using globalCounterId
        const std::pair<uint16_t, armnn::BackendId>& backendCounterIdPair =
                m_ProfilingService.GetCounterMappings().GetBackendId(globalCounterId);
        if (backendCounterIdPair.second == m_BackendId)
        {
            activeCounterIds.emplace_back(backendCounterIdPair.first,
                                          globalCounterId,
                                          true,
                                          captureData.GetCapturePeriod());
        }
    }

    return activeCounterIds;
}

bool BackendProfiling::IsProfilingEnabled() const
{
    return m_ProfilingService.IsProfilingEnabled();
}

}    // namespace profiling
}    // namespace armnn
