//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <string>

namespace armnn
{

namespace gatordmock
{


/**
 * A class that implements a Mock Gatord server. It will listen on a specified Unix domain socket (UDS)
 * namespace for client connections.
 */
class GatordMockService
{
public:

    /**
     * Establish the Unix domain socket and set it to listen for connections.
     *
     * @param udsNamespace the namespace (socket address) associated with the listener.
     * @return true only if the socket has been correctly setup.
     */
    bool OpenListeningSocket(std::string udsNamespace);

    /**
     * Block waiting to accept one client to connect to the UDS.
     *
     * @return the file descriptor of the client connection.
     */
    int BlockForOneClient();

private:

    int m_ListeningSocket;
};


} // namespace gatordmock

} // namespace armnn


