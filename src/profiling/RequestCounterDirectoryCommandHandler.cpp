//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RequestCounterDirectoryCommandHandler.hpp"

namespace armnn
{

namespace profiling
{

void RequestCounterDirectoryCommandHandler::operator()(const Packet& packet)
{
    BOOST_ASSERT(packet.GetLength() == 0);

    // Write packet to Counter Stream Buffer
    m_SendCounterPacket.SendCounterDirectoryPacket(m_CounterDirectory);
}

} // namespace profiling

} // namespace armnn