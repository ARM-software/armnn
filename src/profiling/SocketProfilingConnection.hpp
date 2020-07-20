//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "IProfilingConnection.hpp"

#include <Runtime.hpp>
#include <common/include/NetworkSockets.hpp>

#pragma once

namespace armnn
{
namespace profiling
{

class SocketProfilingConnection : public IProfilingConnection
{
public:
    SocketProfilingConnection();
    bool IsOpen() const final;
    void Close() final;
    bool WritePacket(const unsigned char* buffer, uint32_t length) final;
    arm::pipe::Packet ReadPacket(uint32_t timeout) final;

private:

    // Read a full packet from the socket.
    arm::pipe::Packet ReceivePacket();

#ifndef __APPLE__
    // To indicate we want to use an abstract UDS ensure the first character of the address is 0.
    const char* m_GatorNamespace = "\0gatord_namespace";
#else
    // MACOSX does not support abstract UDS
    const char* m_GatorNamespace = "/tmp/gatord_namespace";
#endif
    arm::pipe::PollFd m_Socket[1]{};
};

} // namespace profiling
} // namespace armnn
