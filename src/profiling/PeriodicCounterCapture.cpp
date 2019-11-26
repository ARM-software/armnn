//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PeriodicCounterCapture.hpp"

#include <armnn/Logging.hpp>

#include <iostream>

namespace armnn
{

namespace profiling
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
    m_PeriodCaptureThread = std::thread(&PeriodicCounterCapture::Capture, this, std::ref(m_ReadCounterValues));
}

void PeriodicCounterCapture::Stop()
{
    // Signal the capture thread to stop
    m_KeepRunning.store(false);

    // Check that the capture thread is running
    if (m_PeriodCaptureThread.joinable())
    {
        // Wait for the capture thread to complete operations
        m_PeriodCaptureThread.join();
    }

    // Mark the capture thread as not running
    m_IsRunning = false;
}

CaptureData PeriodicCounterCapture::ReadCaptureData()
{
    return m_CaptureDataHolder.GetCaptureData();
}

void PeriodicCounterCapture::Capture(const IReadCounterValues& readCounterValues)
{
    do
    {
        // Check if the current capture data indicates that there's data capture
        auto currentCaptureData = ReadCaptureData();
        const std::vector<uint16_t>& counterIds = currentCaptureData.GetCounterIds();

        if (currentCaptureData.GetCapturePeriod() == 0 || counterIds.empty())
        {
            // No data capture, wait the indicated capture period (milliseconds)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        std::vector<std::pair<uint16_t, uint32_t>> values;
        auto numCounters = counterIds.size();
        values.reserve(numCounters);

        // Create a vector of pairs of CounterIndexes and Values
        for (uint16_t index = 0; index < numCounters; ++index)
        {
            auto requestedId = counterIds[index];
            uint32_t counterValue = 0;
            try
            {
                counterValue = readCounterValues.GetCounterValue(requestedId);
            }
            catch (const Exception& e)
            {
                // Report the error and continue
                ARMNN_LOG(warning) << "An error has occurred when getting a counter value: "
                                           << e.what();
                continue;
            }
            values.emplace_back(std::make_pair(requestedId, counterValue));
        }

        // Take a timestamp
        uint64_t timestamp = GetTimestamp();

        // Write a Periodic Counter Capture packet to the Counter Stream Buffer
        m_SendCounterPacket.SendPeriodicCounterCapturePacket(timestamp, values);

        // Notify the Send Thread that new data is available in the Counter Stream Buffer
        m_SendCounterPacket.SetReadyToRead();

        // Wait the indicated capture period (microseconds)
        std::this_thread::sleep_for(std::chrono::microseconds(currentCaptureData.GetCapturePeriod()));

    }
    while (m_KeepRunning.load());
}

} // namespace profiling

} // namespace armnn
