//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <Packet.hpp>
#include <CommandHandlerFunctor.hpp>

#include <vector>

namespace armnn
{

namespace gatordmock
{

struct CounterCaptureValues
{
    uint64_t m_Timestamp;
    std::vector<uint16_t> m_Uids;
    std::vector<uint32_t> m_Values;
};

class PeriodicCounterCaptureCommandHandler : public profiling::CommandHandlerFunctor
{

public:
    PeriodicCounterCaptureCommandHandler(uint32_t packetId,
                                         uint32_t version,
                                         bool echoPackets)
        : CommandHandlerFunctor(packetId, version)
        , m_EchoPackets(echoPackets)
    {}

    void operator()(const armnn::profiling::Packet& packet) override;

    CounterCaptureValues m_CounterCaptureValues;

    uint64_t m_CurrentPeriodValue = 0;

private:
    void ParseData(const armnn::profiling::Packet& packet);

    uint64_t m_FirstTimestamp = 0, m_SecondTimestamp = 0;

    bool m_HeaderPrinted = false;
    bool m_EchoPackets;
};

} // namespace gatordmock

} // namespace armnn
