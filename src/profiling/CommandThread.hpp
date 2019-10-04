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
                  IProfilingConnection& socketProfilingConnection)
        : m_Timeout(timeout)
        , m_StopAfterTimeout(stopAfterTimeout)
        , m_IsRunning(false)
        , m_KeepRunning(false)
        , m_CommandThread()
        , m_CommandHandlerRegistry(commandHandlerRegistry)
        , m_PacketVersionResolver(packetVersionResolver)
        , m_SocketProfilingConnection(socketProfilingConnection)
    {}
    ~CommandThread() { Stop(); }

    void Start();
    void Stop();

    bool IsRunning() const;

    void SetTimeout(uint32_t timeout);
    void SetStopAfterTimeout(bool stopAfterTimeout);

private:
    void WaitForPacket();

    std::atomic<uint32_t> m_Timeout;
    std::atomic<bool> m_StopAfterTimeout;
    std::atomic<bool> m_IsRunning;
    std::atomic<bool> m_KeepRunning;
    std::thread m_CommandThread;

    CommandHandlerRegistry& m_CommandHandlerRegistry;
    PacketVersionResolver& m_PacketVersionResolver;
    IProfilingConnection& m_SocketProfilingConnection;
};

} // namespace profiling

} // namespace armnn
