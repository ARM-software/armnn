//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IBufferManager.hpp"
#include "IConsumer.hpp"
#include "ICounterDirectory.hpp"
#include "ISendCounterPacket.hpp"
#include "ISendThread.hpp"
#include "IProfilingConnection.hpp"
#include "ProfilingStateMachine.hpp"
#include "ProfilingUtils.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <type_traits>

namespace armnn
{

namespace profiling
{

class SendThread : public ISendThread, public IConsumer
{
public:
    SendThread(ProfilingStateMachine& profilingStateMachine,
        IBufferManager& buffer, ISendCounterPacket& sendCounterPacket, int timeout= 1000);
    ~SendThread()
    {
        // Don't rethrow when destructing the object
        Stop(false);
    }
    void Start(IProfilingConnection& profilingConnection) override;

    void Stop(bool rethrowSendThreadExceptions = true) override;

    void SetReadyToRead() override;

    bool IsRunning() { return m_IsRunning.load(); }

    bool WaitForPacketSent(uint32_t timeout);

private:
    void Send(IProfilingConnection& profilingConnection);

    void FlushBuffer(IProfilingConnection& profilingConnection, bool notifyWatchers = true);

    ProfilingStateMachine& m_StateMachine;
    IBufferManager& m_BufferManager;
    ISendCounterPacket& m_SendCounterPacket;
    int m_Timeout;
    std::mutex m_WaitMutex;
    std::condition_variable m_WaitCondition;
    std::thread m_SendThread;
    std::atomic<bool> m_IsRunning;
    std::atomic<bool> m_KeepRunning;
    // m_ReadyToRead will be protected by m_WaitMutex
    bool m_ReadyToRead;
    // m_PacketSent will be protected by m_PacketSentWaitMutex
    bool m_PacketSent;
    std::exception_ptr m_SendThreadException;
    std::mutex m_PacketSentWaitMutex;
    std::condition_variable m_PacketSentWaitCondition;

};

} // namespace profiling

} // namespace armnn
