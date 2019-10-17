//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Packet.hpp"
#include "CommandHandlerFunctor.hpp"
#include "Holder.hpp"
#include "SendCounterPacket.hpp"
#include "IPeriodicCounterCapture.hpp"
#include "ICounterValues.hpp"

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
                                           Holder& captureDataHolder,
                                           IPeriodicCounterCapture& periodicCounterCapture,
                                           const IReadCounterValues& readCounterValue,
                                           ISendCounterPacket& sendCounterPacket,
                                           const ProfilingStateMachine& profilingStateMachine)
        : CommandHandlerFunctor(familyId, packetId, version)
        , m_CaptureDataHolder(captureDataHolder)
        , m_PeriodicCounterCapture(periodicCounterCapture)
        , m_ReadCounterValues(readCounterValue)
        , m_SendCounterPacket(sendCounterPacket)
        , m_StateMachine(profilingStateMachine)
    {}

    void operator()(const Packet& packet) override;

private:
    Holder& m_CaptureDataHolder;
    IPeriodicCounterCapture& m_PeriodicCounterCapture;
    const IReadCounterValues& m_ReadCounterValues;
    ISendCounterPacket& m_SendCounterPacket;
    const ProfilingStateMachine& m_StateMachine;

    void ParseData(const Packet& packet, CaptureData& captureData);
};

} // namespace profiling

} // namespace armnn

