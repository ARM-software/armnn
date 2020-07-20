//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once


#include <armnn/utility/IgnoreUnused.hpp>

#include <cstdint>
#include <memory>
#include <vector>

// forward declare to prevent a circular dependency
namespace arm
{
namespace pipe
{
    class Packet;
} // namespace pipe
} // namespace arm

namespace armnn
{

namespace profiling
{

enum class TargetEndianness
{
    BeWire,
    LeWire
};

// the handlers need to be able to do two
// things to service the FileOnlyProfilingConnection
// and any other implementation of IProfilingConnection
// set the endianness and write a packet back i.e.
// return a packet and close the connection
class IInternalProfilingConnection
{
public:
    virtual ~IInternalProfilingConnection() {};

    virtual void SetEndianess(const TargetEndianness& endianness) = 0;

    virtual void ReturnPacket(arm::pipe::Packet& packet) = 0;

    virtual void Close() = 0;
};

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
    virtual void HandlePacket(const arm::pipe::Packet& packet) = 0;

    /// Set a profiling connection on the handler. Only need to implement this
    /// function if the handler will be writing data back to the profiled application.
    virtual void SetConnection(IInternalProfilingConnection* profilingConnection)
    {armnn::IgnoreUnused(profilingConnection);}
};

using ILocalPacketHandlerPtr = std::unique_ptr<ILocalPacketHandler>;
using ILocalPacketHandlerSharedPtr = std::shared_ptr<ILocalPacketHandler>;

} // namespace profiling

} // namespace armnn