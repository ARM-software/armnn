//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CommandHandlerRegistry.hpp"
#include "IProfilingConnection.hpp"
#include "PacketVersionResolver.hpp"

#include <atomic>
#include <thread>

namespace armnn
{

namespace profiling
{

class CommandHandler
{
public:
    CommandHandler(uint32_t timeout,
                  bool stopAfterTimeout,
                  CommandHandlerRegistry& commandHandlerRegistry,
                  PacketVersionResolver& packetVersionResolver)
        : m_Timeout(timeout)
        , m_StopAfterTimeout(stopAfterTimeout)
        , m_IsRunning(false)
        , m_KeepRunning(false)
        , m_CommandThread()
        , m_CommandHandlerRegistry(commandHandlerRegistry)
        , m_PacketVersionResolver(packetVersionResolver)
    {}
    ~CommandHandler() { Stop(); }

    void SetTimeout(uint32_t timeout) { m_Timeout.store(timeout); }
    void SetStopAfterTimeout(bool stopAfterTimeout) { m_StopAfterTimeout.store(stopAfterTimeout); }

    void Start(IProfilingConnection& profilingConnection);
    void Stop();
    bool IsRunning() const { return m_IsRunning.load(); }

private:
    void HandleCommands(IProfilingConnection& profilingConnection);

    std::atomic<uint32_t> m_Timeout;
    std::atomic<bool> m_StopAfterTimeout;
    std::atomic<bool> m_IsRunning;
    std::atomic<bool> m_KeepRunning;
    std::thread m_CommandThread;

    CommandHandlerRegistry& m_CommandHandlerRegistry;
    PacketVersionResolver& m_PacketVersionResolver;
};

} // namespace profiling

} // namespace armnn
