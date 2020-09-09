//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PeriodicCounterSelectionCommandHandler.hpp"
#include "ProfilingUtils.hpp"

#include <armnn/Types.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <fmt/format.h>

#include <vector>

namespace armnn
{

namespace profiling
{

void PeriodicCounterSelectionCommandHandler::ParseData(const arm::pipe::Packet& packet, CaptureData& captureData)
{
    std::vector<uint16_t> counterIds;
    uint32_t sizeOfUint32 = armnn::numeric_cast<uint32_t>(sizeof(uint32_t));
    uint32_t sizeOfUint16 = armnn::numeric_cast<uint32_t>(sizeof(uint16_t));
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

void PeriodicCounterSelectionCommandHandler::operator()(const arm::pipe::Packet& packet)
{
    ProfilingState currentState = m_StateMachine.GetCurrentState();
    switch (currentState)
    {
    case ProfilingState::Uninitialised:
    case ProfilingState::NotConnected:
    case ProfilingState::WaitingForAck:
        throw RuntimeException(fmt::format("Periodic Counter Selection Command Handler invoked while in "
                                           "an wrong state: {}",
                                           GetProfilingStateName(currentState)));
    case ProfilingState::Active:
    {
        // Process the packet
        if (!(packet.GetPacketFamily() == 0u && packet.GetPacketId() == 4u))
        {
            throw armnn::InvalidArgumentException(fmt::format("Expected Packet family = 0, id = 4 but "
                                                              "received family = {}, id = {}",
                                                              packet.GetPacketFamily(),
                                                              packet.GetPacketId()));
        }

        // Parse the packet to get the capture period and counter UIDs
        CaptureData captureData;
        ParseData(packet, captureData);

        // Get the capture data
        uint32_t capturePeriod = captureData.GetCapturePeriod();
        // Validate that the capture period is within the acceptable range.
        if (capturePeriod > 0  && capturePeriod < LOWEST_CAPTURE_PERIOD)
        {
            capturePeriod = LOWEST_CAPTURE_PERIOD;
        }
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
            validCounterIds.emplace_back(counterId);
        }

        std::sort(validCounterIds.begin(), validCounterIds.end());

        auto backendIdStart = std::find_if(validCounterIds.begin(), validCounterIds.end(), [&](uint16_t& counterId)
        {
            return counterId > m_MaxArmCounterId;
        });

        std::set<armnn::BackendId> activeBackends;
        std::set<uint16_t> backendCounterIds = std::set<uint16_t>(backendIdStart, validCounterIds.end());

        if (m_BackendCounterMap.size() != 0)
        {
            std::set<uint16_t> newCounterIds;
            std::set<uint16_t> unusedCounterIds;

            // Get any backend counter ids that is in backendCounterIds but not in m_PrevBackendCounterIds
            std::set_difference(backendCounterIds.begin(), backendCounterIds.end(),
                                m_PrevBackendCounterIds.begin(), m_PrevBackendCounterIds.end(),
                                std::inserter(newCounterIds, newCounterIds.begin()));

            // Get any backend counter ids that is in m_PrevBackendCounterIds but not in backendCounterIds
            std::set_difference(m_PrevBackendCounterIds.begin(), m_PrevBackendCounterIds.end(),
                                backendCounterIds.begin(), backendCounterIds.end(),
                                std::inserter(unusedCounterIds, unusedCounterIds.begin()));

            activeBackends = ProcessBackendCounterIds(capturePeriod, newCounterIds, unusedCounterIds);
        }
        else
        {
            activeBackends = ProcessBackendCounterIds(capturePeriod, backendCounterIds, {});
        }

        // save the new backend counter ids for next time
        m_PrevBackendCounterIds = backendCounterIds;

        // Set the capture data with only the valid armnn counter UIDs
        m_CaptureDataHolder.SetCaptureData(capturePeriod, {validCounterIds.begin(), backendIdStart}, activeBackends);

        // Echo back the Periodic Counter Selection packet to the Counter Stream Buffer
        m_SendCounterPacket.SendPeriodicCounterSelectionPacket(capturePeriod, validCounterIds);

        if (capturePeriod == 0 || validCounterIds.empty())
        {
            // No data capture stop the thread
            m_PeriodicCounterCapture.Stop();
        }
        else
        {
            // Start the Period Counter Capture thread (if not running already)
            m_PeriodicCounterCapture.Start();
        }

        break;
    }
    default:
        throw RuntimeException(fmt::format("Unknown profiling service state: {}",
                                           static_cast<int>(currentState)));
    }
}

std::set<armnn::BackendId> PeriodicCounterSelectionCommandHandler::ProcessBackendCounterIds(
                                                                      const uint32_t capturePeriod,
                                                                      const std::set<uint16_t> newCounterIds,
                                                                      const std::set<uint16_t> unusedCounterIds)
{
    std::set<armnn::BackendId> changedBackends;
    std::set<armnn::BackendId> activeBackends = m_CaptureDataHolder.GetCaptureData().GetActiveBackends();

    for (uint16_t counterId : newCounterIds)
    {
        auto backendId = m_CounterIdMap.GetBackendId(counterId);
        m_BackendCounterMap[backendId.second].emplace_back(backendId.first);
        changedBackends.insert(backendId.second);
    }
    // Add any new backends to active backends
    activeBackends.insert(changedBackends.begin(), changedBackends.end());

    for (uint16_t counterId : unusedCounterIds)
    {
        auto backendId = m_CounterIdMap.GetBackendId(counterId);
        std::vector<uint16_t>& backendCounters = m_BackendCounterMap[backendId.second];

        backendCounters.erase(std::remove(backendCounters.begin(), backendCounters.end(), backendId.first));

        if(backendCounters.size() == 0)
        {
            // If a backend has no counters associated with it we remove it from active backends and
            // send a capture period of zero with an empty vector, this will deactivate all the backends counters
            activeBackends.erase(backendId.second);
            ActivateBackedCounters(backendId.second, 0, {});
        }
        else
        {
            changedBackends.insert(backendId.second);
        }
    }

    // If the capture period remains the same we only need to update the backends who's counters have changed
    if(capturePeriod == m_PrevCapturePeriod)
    {
        for (auto backend : changedBackends)
        {
            ActivateBackedCounters(backend, capturePeriod, m_BackendCounterMap[backend]);
        }
    }
    // Otherwise update all the backends with the new capture period and any new/unused counters
    else
    {
        for (auto backend : m_BackendCounterMap)
        {
            ActivateBackedCounters(backend.first, capturePeriod, backend.second);
        }
        if(capturePeriod == 0)
        {
            activeBackends = {};
        }
        m_PrevCapturePeriod = capturePeriod;
    }

    return activeBackends;
}

} // namespace profiling

} // namespace armnn
