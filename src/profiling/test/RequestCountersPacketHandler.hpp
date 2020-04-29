//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>
#include <armnn/profiling/ILocalPacketHandler.hpp>
#include "Packet.hpp"
#include "ProfilingUtils.hpp"

namespace armnn
{

namespace profiling
{

class RequestCountersPacketHandler : public ILocalPacketHandler
{
public:
    explicit RequestCountersPacketHandler(uint32_t capturePeriod = LOWEST_CAPTURE_PERIOD) :
        m_CapturePeriod(capturePeriod),
        m_Connection(nullptr),
        m_CounterDirectoryMessageHeader(ConstructHeader(0, 2)) {}

    std::vector<uint32_t> GetHeadersAccepted() override; // ILocalPacketHandler

    void HandlePacket(const Packet& packet) override; // ILocalPacketHandler

    void SetConnection(IInternalProfilingConnection* profilingConnection) override // ILocalPacketHandler
    {
        m_Connection = profilingConnection;
    }

private:
    uint32_t m_CapturePeriod;
    IInternalProfilingConnection* m_Connection;
    uint32_t m_CounterDirectoryMessageHeader;
    std::vector<uint16_t> m_IdList;

    void SendCounterSelectionPacket();
};

} // namespace profiling

} // namespace armnn