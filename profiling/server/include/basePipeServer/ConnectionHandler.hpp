//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BasePipeServer.hpp"
#include <string>

namespace arm
{

namespace pipe
{

class ConnectionHandler
{
public:
    /// Constructor establishes the Unix domain socket and sets it to listen for connections.
    /// @param udsNamespace the namespace (socket address) associated with the listener.
    /// @throws SocketConnectionException if the socket has been incorrectly setup.
    ConnectionHandler(const std::string& udsNamespace, const bool setNonBlocking);

    ~ConnectionHandler()
    {
        // We have set SOCK_CLOEXEC on this socket but we'll close it to be good citizens.
        arm::pipe::Close(m_ListeningSocket);
    }

    ConnectionHandler(const ConnectionHandler&) = delete;
    ConnectionHandler& operator=(const ConnectionHandler&) = delete;

    ConnectionHandler(ConnectionHandler&&) = delete;
    ConnectionHandler& operator=(ConnectionHandler&&) = delete;

    /// Attempt to open a new socket to the client and use it to construct a new basePipeServer
    /// @param echoPackets if true the raw packets will be printed to stdout.
    /// @return if successful a unique_ptr to a basePipeServer otherwise a nullptr
    std::unique_ptr<BasePipeServer> GetNewBasePipeServer(const bool echoPackets);

private:

    arm::pipe::Socket m_ListeningSocket;
};

} // namespace pipe
} // namespace arm
