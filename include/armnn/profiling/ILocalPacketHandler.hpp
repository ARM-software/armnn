//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once


#include <armnn/utility/IgnoreUnused.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace armnn
{

namespace profiling
{
// forward declare to prevent a circular dependency
class Packet;
class IProfilingConnection;

class ILocalPacketHandler
{
public:
    virtual ~ILocalPacketHandler() {};

    /// @return lists the headers of the packets that this handler accepts
    ///         only these packets will get sent to this handler.
    ///         If this function returns an empty list then ALL packets
    ///         will be sent to the PacketHandler i.e. a universal handler.
    virtual std::vector<uint32_t> GetHeadersAccepted() = 0;

    /// process the packet
    virtual void HandlePacket(const Packet& packet) = 0;

    /// Set a profiling connection on the handler. Only need to implement this
    /// function if the handler will be writing data back to the profiled application.
    virtual void SetConnection(IProfilingConnection* profilingConnection) {armnn::IgnoreUnused(profilingConnection);}
};

using ILocalPacketHandlerPtr = std::unique_ptr<ILocalPacketHandler>;
using ILocalPacketHandlerSharedPtr = std::shared_ptr<ILocalPacketHandler>;

} // namespace profiling

} // namespace armnn