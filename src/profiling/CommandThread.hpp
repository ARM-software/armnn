//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CommandHandlerRegistry.hpp"
#include "IProfilingConnection.hpp"
#include "PacketVersionResolver.hpp"
#include "ProfilingService.hpp"

#include <atomic>
#include <thread>

namespace armnn
{

namespace profiling
{

class CommandThread
{
public:
    CommandThread(uint32_t timeout,
                  bool stopAfterTimeout,
                  CommandHandlerRegistry& commandHandlerRegistry,
                  PacketVersionResolver& packetVersionResolver,
                  IProfilingConnection& socketProfilingConnection);

    void Start();
    void Stop();
    void Join();
    bool IsRunning() const;
    bool StopAfterTimeout(bool StopAfterTimeout);

private:
    void WaitForPacket();

    uint32_t m_Timeout;
    bool m_StopAfterTimeout;
    std::atomic<bool> m_IsRunning;
    std::atomic<bool> m_KeepRunning;
    std::thread m_CommandThread;

    CommandHandlerRegistry& m_CommandHandlerRegistry;
    PacketVersionResolver& m_PacketVersionResolver;
    IProfilingConnection& m_SocketProfilingConnection;
};

}//namespace profiling

}//namespace armnn