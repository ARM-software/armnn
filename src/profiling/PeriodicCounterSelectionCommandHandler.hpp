//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CounterIdMap.hpp"
#include "Packet.hpp"
#include "CommandHandlerFunctor.hpp"
#include "Holder.hpp"
#include "ProfilingStateMachine.hpp"
#include "SendCounterPacket.hpp"
#include "IPeriodicCounterCapture.hpp"
#include "ICounterValues.hpp"

#include "armnn/backends/profiling/IBackendProfilingContext.hpp"
#include "armnn/Logging.hpp"
#include "armnn/BackendRegistry.hpp"

#include <set>


namespace armnn
{

namespace profiling
{


class PeriodicCounterSelectionCommandHandler : public CommandHandlerFunctor
{

public:
    PeriodicCounterSelectionCommandHandler(uint32_t familyId,
                                           uint32_t packetId,
                                           uint32_t version,
                                           const std::unordered_map<BackendId,
                                                   std::shared_ptr<armnn::profiling::IBackendProfilingContext>>&
                                           backendProfilingContext,
                                           const ICounterMappings& counterIdMap,
                                           Holder& captureDataHolder,
                                           const uint16_t maxArmnnCounterId,
                                           IPeriodicCounterCapture& periodicCounterCapture,
                                           const IReadCounterValues& readCounterValue,
                                           ISendCounterPacket& sendCounterPacket,
                                           const ProfilingStateMachine& profilingStateMachine)
        : CommandHandlerFunctor(familyId, packetId, version)
        , m_BackendProfilingContext(backendProfilingContext)
        , m_CounterIdMap(counterIdMap)
        , m_CaptureDataHolder(captureDataHolder)
        , m_MaxArmCounterId(maxArmnnCounterId)
        , m_PeriodicCounterCapture(periodicCounterCapture)
        , m_PrevCapturePeriod(0)
        , m_ReadCounterValues(readCounterValue)
        , m_SendCounterPacket(sendCounterPacket)
        , m_StateMachine(profilingStateMachine)

    {

    }

    void operator()(const Packet& packet) override;

private:

    std::unordered_map<armnn::BackendId, std::vector<uint16_t>> m_BackendCounterMap;
    const std::unordered_map<BackendId,
          std::shared_ptr<armnn::profiling::IBackendProfilingContext>>& m_BackendProfilingContext;
    const ICounterMappings& m_CounterIdMap;
    Holder& m_CaptureDataHolder;
    const uint16_t m_MaxArmCounterId;
    IPeriodicCounterCapture& m_PeriodicCounterCapture;
    uint32_t m_PrevCapturePeriod;
    std::set<uint16_t> m_PrevBackendCounterIds;
    const IReadCounterValues& m_ReadCounterValues;
    ISendCounterPacket& m_SendCounterPacket;
    const ProfilingStateMachine& m_StateMachine;

    void ActivateBackedCounters(const armnn::BackendId backendId,
                                const uint32_t capturePeriod,
                                const std::vector<uint16_t> counterIds)
    {
        Optional<std::string> errorMsg =
                m_BackendProfilingContext.at(backendId)->ActivateCounters(capturePeriod, counterIds);

        if(errorMsg.has_value())
        {
            ARMNN_LOG(warning) << "An error has occurred when activating counters of " << backendId << ": "
                               << errorMsg.value();
        }
    }
    void ParseData(const Packet& packet, CaptureData& captureData);
    std::set<armnn::BackendId> ProcessBackendCounterIds(const u_int32_t capturePeriod,
                                                        std::set<uint16_t> newCounterIds,
                                                        std::set<uint16_t> unusedCounterIds);

};

} // namespace profiling

} // namespace armnn

