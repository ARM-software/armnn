//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IBufferWrapper.hpp"
#include "ISendCounterPacket.hpp"
#include "ICounterDirectory.hpp"
#include "IProfilingConnection.hpp"

#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>

namespace armnn
{

namespace profiling
{

class SendCounterPacket : public ISendCounterPacket
{
public:
    using CategoryRecord   = std::vector<uint32_t>;
    using DeviceRecord     = std::vector<uint32_t>;
    using CounterSetRecord = std::vector<uint32_t>;
    using EventRecord      = std::vector<uint32_t>;

    using IndexValuePairsVector = std::vector<std::pair<uint16_t, uint32_t>>;

    SendCounterPacket(IProfilingConnection& profilingConnection, IBufferWrapper& buffer)
        : m_ProfilingConnection(profilingConnection)
        , m_Buffer(buffer)
        , m_IsRunning(false)
        , m_KeepRunning(false)
    {}
    ~SendCounterPacket() { Stop(); }

    void SendStreamMetaDataPacket() override;

    void SendCounterDirectoryPacket(const ICounterDirectory& counterDirectory) override;

    void SendPeriodicCounterCapturePacket(uint64_t timestamp, const IndexValuePairsVector& values) override;

    void SendPeriodicCounterSelectionPacket(uint32_t capturePeriod,
                                            const std::vector<uint16_t>& selectedCounterIds) override;

    void SetReadyToRead() override;

    static const unsigned int PIPE_MAGIC = 0x45495434;
    static const unsigned int MAX_METADATA_PACKET_LENGTH = 4096;

    void Start();
    void Stop();
    bool IsRunning() { return m_IsRunning.load(); }

private:
    void Send();

    template <typename ExceptionType>
    void CancelOperationAndThrow(const std::string& errorMessage)
    {
        // Cancel the operation
        m_Buffer.Commit(0);

        // Throw a runtime exception with the given error message
        throw ExceptionType(errorMessage);
    }

    IProfilingConnection& m_ProfilingConnection;
    IBufferWrapper& m_Buffer;
    std::mutex m_WaitMutex;
    std::condition_variable m_WaitCondition;
    std::thread m_SendThread;
    std::atomic<bool> m_IsRunning;
    std::atomic<bool> m_KeepRunning;

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
