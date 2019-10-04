//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PeriodicCounterCapture.hpp"

namespace armnn
{

namespace profiling
{

void PeriodicCounterCapture::Start()
{
    // Check if the capture thread is already running
    if (m_IsRunning.load())
    {
        // The capture thread is already running
        return;
    }

    // Mark the capture thread as running
    m_IsRunning.store(true);

    // Keep the capture procedure going until the capture thread is signalled to stop
    m_KeepRunning.store(true);

    // Start the new capture thread.
    m_PeriodCaptureThread = std::thread(&PeriodicCounterCapture::Capture,
                                        this,
                                        std::ref(m_ReadCounterValues));
}

void PeriodicCounterCapture::Stop()
{
    m_KeepRunning.store(false);

    if (m_PeriodCaptureThread.joinable())
    {
        m_PeriodCaptureThread.join();
    }
}

CaptureData PeriodicCounterCapture::ReadCaptureData()
{
    return m_CaptureDataHolder.GetCaptureData();
}

void PeriodicCounterCapture::Capture(const IReadCounterValues& readCounterValues)
{
    while (m_KeepRunning.load())
    {
        auto currentCaptureData = ReadCaptureData();
        std::vector<uint16_t> counterIds = currentCaptureData.GetCounterIds();
        if (currentCaptureData.GetCapturePeriod() == 0 || counterIds.empty())
        {
            m_KeepRunning.store(false);
            break;
        }

        std::vector<std::pair<uint16_t, uint32_t>> values;
        auto numCounters = counterIds.size();
        values.reserve(numCounters);

        // Create vector of pairs of CounterIndexes and Values
        uint32_t counterValue = 0;
        for (uint16_t index = 0; index < numCounters; ++index)
        {
            auto requestedId = counterIds[index];
            counterValue = readCounterValues.GetCounterValue(requestedId);
            values.emplace_back(std::make_pair(requestedId, counterValue));
        }

        #if USE_CLOCK_MONOTONIC_RAW
            using clock = MonotonicClockRaw;
        #else
            using clock = std::chrono::steady_clock;
        #endif

        // Take a timestamp
        auto timestamp = clock::now();

        m_SendCounterPacket.SendPeriodicCounterCapturePacket(
                    static_cast<uint64_t>(timestamp.time_since_epoch().count()), values);
        std::this_thread::sleep_for(std::chrono::milliseconds(currentCaptureData.GetCapturePeriod()));
    }

    m_IsRunning.store(false);
}

} // namespace profiling

} // namespace armnn
