//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IBufferManager.hpp"
#include "ICounterDirectory.hpp"
#include "ISendCounterPacket.hpp"
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

class SendCounterPacket : public ISendCounterPacket
{
public:
    using CategoryRecord        = std::vector<uint32_t>;
    using DeviceRecord          = std::vector<uint32_t>;
    using CounterSetRecord      = std::vector<uint32_t>;
    using EventRecord           = std::vector<uint32_t>;
    using IndexValuePairsVector = std::vector<std::pair<uint16_t, uint32_t>>;

    SendCounterPacket(ProfilingStateMachine& profilingStateMachine, IBufferManager& buffer, int timeout = 1000)
        : m_StateMachine(profilingStateMachine)
        , m_BufferManager(buffer)
        , m_Timeout(timeout)
        , m_IsRunning(false)
        , m_KeepRunning(false)
        , m_SendThreadException(nullptr)
    {}
    ~SendCounterPacket()
    {
        // Don't rethrow when destructing the object
        Stop(false);
    }

    void SendStreamMetaDataPacket() override;

    void SendCounterDirectoryPacket(const ICounterDirectory& counterDirectory) override;

    void SendPeriodicCounterCapturePacket(uint64_t timestamp, const IndexValuePairsVector& values) override;

    void SendPeriodicCounterSelectionPacket(uint32_t capturePeriod,
                                            const std::vector<uint16_t>& selectedCounterIds) override;

    void SetReadyToRead() override;

    static const unsigned int PIPE_MAGIC = 0x45495434;

    void Start(IProfilingConnection& profilingConnection);
    void Stop(bool rethrowSendThreadExceptions = true);
    bool IsRunning() { return m_IsRunning.load(); }

    void WaitForPacketSent(uint32_t timeout = 1000)
    {
        std::unique_lock<std::mutex> lock(m_PacketSentWaitMutex);
        // Blocks until notified that at least a packet has been sent or until timeout expires.
        m_PacketSentWaitCondition.wait_for(lock, std::chrono::milliseconds(timeout));
    }

private:
    void Send(IProfilingConnection& profilingConnection);

    template <typename ExceptionType>
    void CancelOperationAndThrow(const std::string& errorMessage)
    {
        // Throw a runtime exception with the given error message
        throw ExceptionType(errorMessage);
    }

    template <typename ExceptionType>
    void CancelOperationAndThrow(IPacketBufferPtr& writerBuffer, const std::string& errorMessage)
    {
        if (std::is_same<ExceptionType, armnn::profiling::BufferExhaustion>::value)
        {
            SetReadyToRead();
        }

        if (writerBuffer != nullptr)
        {
            // Cancel the operation
            m_BufferManager.Release(writerBuffer);
        }

        // Throw a runtime exception with the given error message
        throw ExceptionType(errorMessage);
    }

    void FlushBuffer(IProfilingConnection& profilingConnection, bool notifyWatchers = true);

    ProfilingStateMachine& m_StateMachine;
    IBufferManager& m_BufferManager;
    int m_Timeout;
    std::mutex m_WaitMutex;
    std::condition_variable m_WaitCondition;
    std::thread m_SendThread;
    std::atomic<bool> m_IsRunning;
    std::atomic<bool> m_KeepRunning;
    std::exception_ptr m_SendThreadException;
    std::mutex m_PacketSentWaitMutex;
    std::condition_variable m_PacketSentWaitCondition;

protected:
    // Helper methods, protected for testing
    bool CreateCategoryRecord(const CategoryPtr& category,
                              const Counters& counters,
                              CategoryRecord& categoryRecord,
                              std::string& errorMessage);
    bool CreateDeviceRecord(const DevicePtr& device,
                            DeviceRecord& deviceRecord,
                            std::string& errorMessage);
    bool CreateCounterSetRecord(const CounterSetPtr& counterSet,
                                CounterSetRecord& counterSetRecord,
                                std::string& errorMessage);
    bool CreateEventRecord(const CounterPtr& counter,
                           EventRecord& eventRecord,
                           std::string& errorMessage);
};

} // namespace profiling

} // namespace armnn
