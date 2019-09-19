//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "IProfilingConnection.hpp"

#include <poll.h>
#include <Runtime.hpp>

#pragma once

namespace armnn
{
namespace profiling
{

class SocketProfilingConnection : public IProfilingConnection
{
public:
    SocketProfilingConnection();
    bool IsOpen() final;
    void Close() final;
    bool WritePacket(const unsigned char* buffer, uint32_t length) final;
    Packet ReadPacket(uint32_t timeout) final;

private:
    // To indicate we want to use an abstract UDS ensure the first character of the address is 0.
    const char* m_GatorNamespace = "\0gatord_namespace";
    struct pollfd m_Socket[1]{};
};

} // namespace profiling
} // namespace armnn
