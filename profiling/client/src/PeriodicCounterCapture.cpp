//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PeriodicCounterCapture.hpp"

#include <common/include/Logging.hpp>

#include <iostream>

namespace arm
{

namespace pipe
{

void PeriodicCounterCapture::Start()
{
    // Check if the capture thread is already running
    if (m_IsRunning)
    {
        // The capture thread is already running
        return;
    }

    // Mark the capture thread as running
    m_IsRunning = true;

    // Keep the capture procedure going until the capture thread is signalled to stop
    m_KeepRunning.store(true);

    // Start the new capture thread.
#if !defined(ARMNN_DISABLE_THREADS)
    m_PeriodCaptureThread = std::thread(&PeriodicCounterCapture::Capture, this, std::ref(m_ReadCounterValues));
#endif
}

void PeriodicCounterCapture::Stop()
{
    // Signal the capture thread to stop
    m_KeepRunning.store(false);

#if !defined(ARMNN_DISABLE_THREADS)
    // Check that the capture thread is running
    if (m_PeriodCaptureThread.joinable())
    {
        // Wait for the capture thread to complete operations
        m_PeriodCaptureThread.join();
    }
#endif

    // Mark the capture thread as not running
    m_IsRunning = false;
}

CaptureData PeriodicCounterCapture::ReadCaptureData()
{
    return m_CaptureDataHolder.GetCaptureData();
}

void PeriodicCounterCapture::DispatchPeriodicCounterCapturePacket(
    const std::string& backendId, const std::vector<Timestamp>& timestampValues)
{
    // Report counter values
    for (const auto& timestampInfo : timestampValues)
    {
        std::vector<CounterValue> backendCounterValues = timestampInfo.counterValues;
        for_each(backendCounterValues.begin(), backendCounterValues.end(), [&](CounterValue& backendCounterValue)
        {
            // translate the counterId to globalCounterId
            backendCounterValue.counterId = m_CounterIdMap.GetGlobalId(backendCounterValue.counterId, backendId);
        });

        // Send Periodic Counter Capture Packet for the Timestamp
        m_SendCounterPacket.SendPeriodicCounterCapturePacket(timestampInfo.timestamp, backendCounterValues);
    }
}

void PeriodicCounterCapture::Capture(IReadCounterValues& readCounterValues)
{
    do
    {
        // Check if the current capture data indicates that there's data capture
        auto currentCaptureData = ReadCaptureData();
        const std::vector<uint16_t>& counterIds = currentCaptureData.GetCounterIds();
        const uint32_t capturePeriod = currentCaptureData.GetCapturePeriod();

        if (capturePeriod == 0)
        {
            // No data capture, wait the indicated capture period (milliseconds), if it is not zero
#if !defined(ARMNN_DISABLE_THREADS)
            std::this_thread::sleep_for(std::chrono::milliseconds(50u));
#endif
            continue;
        }

        if(counterIds.size() != 0)
        {
            std::vector<CounterValue> counterValues;

            auto numCounters = counterIds.size();
            counterValues.reserve(numCounters);

            // Create a vector of pairs of CounterIndexes and Values
            for (uint16_t index = 0; index < numCounters; ++index)
            {
                auto requestedId = counterIds[index];
                uint32_t counterValue = 0;
                try
                {
                    counterValue = readCounterValues.GetDeltaCounterValue(requestedId);
                }
                catch (const arm::pipe::ProfilingException& e)
                {
                    // Report the error and continue
                    ARM_PIPE_LOG(warning) << "An error has occurred when getting a counter value: "
                                       << e.what();
                    continue;
                }

                counterValues.emplace_back(CounterValue {requestedId, counterValue });
            }

            // Send Periodic Counter Capture Packet for the Timestamp
            m_SendCounterPacket.SendPeriodicCounterCapturePacket(GetTimestamp(), counterValues);
        }

        // Report counter values for each active backend
        auto activeBackends = currentCaptureData.GetActiveBackends();
        for_each(activeBackends.begin(), activeBackends.end(), [&](const std::string& backendId)
        {
            DispatchPeriodicCounterCapturePacket(
                backendId, m_BackendProfilingContexts.at(backendId)->ReportCounterValues());
        });

        // Wait the indicated capture period (microseconds)
#if !defined(ARMNN_DISABLE_THREADS)
        std::this_thread::sleep_for(std::chrono::microseconds(capturePeriod));
#endif
    }
    while (m_KeepRunning.load());
}

} // namespace pipe

} // namespace arm
