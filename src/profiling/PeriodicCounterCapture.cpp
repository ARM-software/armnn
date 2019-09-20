//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PeriodicCounterCapture.hpp"

namespace armnn
{

namespace profiling
{

PeriodicCounterCapture::PeriodicCounterCapture(const Holder& data, ISendCounterPacket& packet,
                                               const IReadCounterValue& readCounterValue)
    : m_CaptureDataHolder(data)
    , m_IsRunning(false)
    , m_ReadCounterValue(readCounterValue)
    , m_SendCounterPacket(packet)
{}

CaptureData PeriodicCounterCapture::ReadCaptureData()
{
    return m_CaptureDataHolder.GetCaptureData();
}

void PeriodicCounterCapture::Functionality(const IReadCounterValue& readCounterValue)
{
    bool threadRunning = true;

    while(threadRunning)
    {
        auto currentCaptureData = ReadCaptureData();
        std::vector<uint16_t> counterIds = currentCaptureData.GetCounterIds();
        if (currentCaptureData.GetCapturePeriod() == 0 || counterIds.empty())
        {
            threadRunning = false;
            m_IsRunning.store(false, std::memory_order_relaxed);
        }
        else
        {
            std::vector<std::pair<uint16_t, uint32_t>> values;
            auto numCounters = counterIds.size();
            values.reserve(numCounters);

            // Create vector of pairs of CounterIndexes and Values
            uint32_t counterValue;
            for (uint16_t index = 0; index < numCounters; ++index)
            {
                auto requestedId = counterIds[index];
                readCounterValue.GetCounterValue(requestedId, counterValue);
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
    }
}

void PeriodicCounterCapture::Start()
{
    bool tstVal = false;

    if (m_IsRunning.compare_exchange_strong(tstVal, true, std::memory_order_relaxed))
    {
        // Check that the thread execution is finished.
        if (m_PeriodCaptureThread.joinable())
        {
            m_PeriodCaptureThread.join();
        }
        // Starts the new thread.
        m_PeriodCaptureThread = std::thread(&PeriodicCounterCapture::Functionality, this,
                                            std::ref(m_ReadCounterValue));
    }
}

void PeriodicCounterCapture::Join()
{
    m_PeriodCaptureThread.join();
}

} // namespace profiling

} // namespace armnn
