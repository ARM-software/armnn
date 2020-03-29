//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PrintPacketHeaderHandler.hpp"

#include <iostream>

namespace armnn
{

namespace profiling
{

std::vector<uint32_t> PrintPacketHeaderHandler::GetHeadersAccepted()
{
    return std::vector<uint32_t>();
}

void PrintPacketHeaderHandler::HandlePacket(const Packet& packet)
{
    std::stringstream ss;
    ss << "Handler Received Outgoing Packet [" << packet.GetPacketFamily() << ":" << packet.GetPacketId() << "]";
    ss << " Length [" << packet.GetLength() << "]" << std::endl;
    std::cout << ss.str() << std::endl;
};

} // namespace profiling

} // namespace armnn