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

#include <vector>
#include <thread>
#include <atomic>

namespace armnn
{

namespace profiling
{

class PeriodicCounterSelectionCommandHandler : public CommandHandlerFunctor
{

public:
    PeriodicCounterSelectionCommandHandler(uint32_t packetId, uint32_t version, Holder& captureDataHolder,
                                           IPeriodicCounterCapture& captureThread,
                                           ISendCounterPacket& sendCounterPacket)
    : CommandHandlerFunctor(packetId, version),
    m_CaptureDataHolder(captureDataHolder),
    m_CaptureThread(captureThread),
    m_SendCounterPacket(sendCounterPacket)
    {}

    void operator()(const Packet& packet) override;


private:
    Holder& m_CaptureDataHolder;
    IPeriodicCounterCapture& m_CaptureThread;
    ISendCounterPacket& m_SendCounterPacket;
    void ParseData(const Packet& packet, CaptureData& captureData);
};

} // namespace profiling

} // namespace armnn

