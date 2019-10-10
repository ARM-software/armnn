//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PeriodicCounterSelectionCommandHandler.hpp"
#include "ProfilingUtils.hpp"

#include <boost/numeric/conversion/cast.hpp>
#include <boost/format.hpp>

#include <vector>

namespace armnn
{

namespace profiling
{

void PeriodicCounterSelectionCommandHandler::ParseData(const Packet& packet, CaptureData& captureData)
{
    std::vector<uint16_t> counterIds;
    uint32_t sizeOfUint32 = boost::numeric_cast<uint32_t>(sizeof(uint32_t));
    uint32_t sizeOfUint16 = boost::numeric_cast<uint32_t>(sizeof(uint16_t));
    uint32_t offset = 0;

    if (packet.GetLength() < 4)
    {
        // Insufficient packet size
        return;
    }

    // Parse the capture period
    uint32_t capturePeriod = ReadUint32(packet.GetData(), offset);

    // Set the capture period
    captureData.SetCapturePeriod(capturePeriod);

    // Parse the counter ids
    unsigned int counters = (packet.GetLength() - 4) / 2;
    if (counters > 0)
    {
        counterIds.reserve(counters);
        offset += sizeOfUint32;
        for (unsigned int i = 0; i < counters; ++i)
        {
            // Parse the counter id
            uint16_t counterId = ReadUint16(packet.GetData(), offset);
            counterIds.emplace_back(counterId);
            offset += sizeOfUint16;
        }
    }

    // Set the counter ids
    captureData.SetCounterIds(counterIds);
}

void PeriodicCounterSelectionCommandHandler::operator()(const Packet& packet)
{
    ProfilingState currentState = m_StateMachine.GetCurrentState();
    switch (currentState)
    {
    case ProfilingState::Uninitialised:
    case ProfilingState::NotConnected:
    case ProfilingState::WaitingForAck:
        throw RuntimeException(boost::str(boost::format("Periodic Counter Selection Command Handler invoked while in "
                                                        "an wrong state: %1%")
                                          % GetProfilingStateName(currentState)));
    case ProfilingState::Active:
    {
        // Process the packet
        if (!(packet.GetPacketFamily() == 0u && packet.GetPacketId() == 4u))
        {
            throw armnn::InvalidArgumentException(boost::str(boost::format("Expected Packet family = 0, id = 4 but "
                                                                           "received family = %1%, id = %2%")
                                                  % packet.GetPacketFamily()
                                                  % packet.GetPacketId()));
        }

        // Parse the packet to get the capture period and counter UIDs
        CaptureData captureData;
        ParseData(packet, captureData);

        // Get the capture data
        const uint32_t capturePeriod = captureData.GetCapturePeriod();
        const std::vector<uint16_t>& counterIds = captureData.GetCounterIds();

        // Check whether the selected counter UIDs are valid
        std::vector<uint16_t> validCounterIds;
        for (uint16_t counterId : counterIds)
        {
            // Check whether the counter is registered
            if (!m_ReadCounterValues.IsCounterRegistered(counterId))
            {
                // Invalid counter UID, ignore it and continue
                continue;
            }

            // The counter is valid
            validCounterIds.push_back(counterId);
        }

        // Set the capture data with only the valid counter UIDs
        m_CaptureDataHolder.SetCaptureData(capturePeriod, validCounterIds);

        // Echo back the Periodic Counter Selection packet to the Counter Stream Buffer
        m_SendCounterPacket.SendPeriodicCounterSelectionPacket(capturePeriod, validCounterIds);

        // Notify the Send Thread that new data is available in the Counter Stream Buffer
        m_SendCounterPacket.SetReadyToRead();

        // Start the Period Counter Capture thread (if not running already)
        m_PeriodicCounterCapture.Start();

        break;
    }
    default:
        throw RuntimeException(boost::str(boost::format("Unknown profiling service state: %1%")
                                          % static_cast<int>(currentState)));
    }
}

} // namespace profiling

} // namespace armnn
