//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/profiling/ILocalPacketHandler.hpp>
#include "Packet.hpp"

namespace armnn
{

namespace profiling
{

class PrintPacketHeaderHandler : public ILocalPacketHandler
{
    virtual std::vector<uint32_t> GetHeadersAccepted();

    virtual void HandlePacket(const Packet& packet);
};

} // namespace profiling

} // namespace armnn
