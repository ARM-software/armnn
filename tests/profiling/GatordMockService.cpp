//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GatordMockService.hpp"

#include <cerrno>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>

namespace armnn
{

namespace gatordmock
{


bool GatordMockService::OpenListeningSocket(std::string udsNamespace)
{
    m_ListeningSocket = socket(PF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (-1 == m_ListeningSocket)
    {
        std::cerr << ": Socket construction failed: " << strerror(errno) << std::endl;
        return false;
    }

    sockaddr_un udsAddress;
    memset(&udsAddress, 0, sizeof(sockaddr_un));
    // We've set the first element of sun_path to be 0, skip over it and copy the namespace after it.
    memcpy(udsAddress.sun_path + 1, udsNamespace.c_str(), strlen(udsNamespace.c_str()));
    udsAddress.sun_family = AF_UNIX;

    // Bind the socket to the UDS namespace.
    if (-1 == bind(m_ListeningSocket, reinterpret_cast<const sockaddr *>(&udsAddress), sizeof(sockaddr_un)))
    {
        std::cerr << ": Binding on socket failed: " << strerror(errno) << std::endl;
        return false;
    }
    // Listen for 1 connection.
    if (-1 == listen(m_ListeningSocket, 1))
    {
        std::cerr << ": Listen call on socket failed: " << strerror(errno) << std::endl;
        return false;
    }
    std::cout << "Bound to UDS namespace: \\0" << udsNamespace << std::endl;
    return true;
}

int GatordMockService::BlockForOneClient()
{
    std::cout << "Waiting for client connection." << std::endl;

    int accepted = accept4(m_ListeningSocket, nullptr, nullptr, SOCK_CLOEXEC);
    if (-1 == accepted)
    {
        std::cerr << ": Failure when waiting for a client connection: " << strerror(errno) << std::endl;
        return -1;
    }

    std::cout << "Client connection established." << std::endl;
    return accepted;
}


} // namespace gatordmock

} // namespace armnn
