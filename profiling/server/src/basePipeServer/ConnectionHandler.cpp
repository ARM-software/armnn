//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ConnectionHandler.hpp"

using namespace armnnUtils;

namespace armnnProfiling
{
ConnectionHandler::ConnectionHandler(const std::string& udsNamespace, const bool setNonBlocking)
{
    Sockets::Initialize();
    m_ListeningSocket = socket(PF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);

    if (-1 == m_ListeningSocket)
    {
        throw SocketConnectionException(": Socket construction failed: ", 1, 1);
    }

    sockaddr_un udsAddress;
    memset(&udsAddress, 0, sizeof(sockaddr_un));
    // We've set the first element of sun_path to be 0, skip over it and copy the namespace after it.
    memcpy(udsAddress.sun_path + 1, udsNamespace.c_str(), strlen(udsNamespace.c_str()));
    udsAddress.sun_family = AF_UNIX;

    // Bind the socket to the UDS namespace.
    if (-1 == bind(m_ListeningSocket, reinterpret_cast<const sockaddr*>(&udsAddress), sizeof(sockaddr_un)))
    {
        throw SocketConnectionException(": Binding on socket failed: ", m_ListeningSocket, errno);
    }
    // Listen for connections.
    if (-1 == listen(m_ListeningSocket, 1))
    {
        throw SocketConnectionException(": Listen call on socket failed: ", m_ListeningSocket, errno);
    }

    if (setNonBlocking)
    {
        Sockets::SetNonBlocking(m_ListeningSocket);
    }
}

std::unique_ptr<BasePipeServer> ConnectionHandler::GetNewBasePipeServer(const bool echoPackets)
{
    armnnUtils::Sockets::Socket clientConnection = armnnUtils::Sockets::Accept(m_ListeningSocket, nullptr, nullptr,
                                                                               SOCK_CLOEXEC);
    if (clientConnection < 1)
    {
        return nullptr;
    }
    return std::make_unique<BasePipeServer>(clientConnection, echoPackets);
}

} // namespace armnnProfiling