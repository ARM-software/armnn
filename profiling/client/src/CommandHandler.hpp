//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IProfilingConnection.hpp"
#include <common/include/PacketVersionResolver.hpp>

#include <common/include/CommandHandlerRegistry.hpp>

#include <atomic>
#if !defined(ARMNN_DISABLE_THREADS)
#include <thread>
#endif

namespace arm
{

namespace pipe
{

class CommandHandler
{
public:
    CommandHandler(uint32_t timeout,
                   bool stopAfterTimeout,
                   arm::pipe::CommandHandlerRegistry& commandHandlerRegistry,
                   arm::pipe::PacketVersionResolver& packetVersionResolver)
        : m_Timeout(timeout),
          m_StopAfterTimeout(stopAfterTimeout),
          m_IsRunning(false),
          m_KeepRunning(false),
#if !defined(ARMNN_DISABLE_THREADS)
          m_CommandThread(),
#endif
          m_CommandHandlerRegistry(commandHandlerRegistry),
          m_PacketVersionResolver(packetVersionResolver)
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
    std::atomic<bool>     m_StopAfterTimeout;
    std::atomic<bool>     m_IsRunning;
    std::atomic<bool>     m_KeepRunning;
#if !defined(ARMNN_DISABLE_THREADS)
    std::thread           m_CommandThread;
#endif

    arm::pipe::CommandHandlerRegistry& m_CommandHandlerRegistry;
    arm::pipe::PacketVersionResolver&  m_PacketVersionResolver;
};

} // namespace pipe

} // namespace arm
