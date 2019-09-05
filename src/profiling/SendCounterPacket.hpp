//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IBufferWrapper.hpp"
#include "ISendCounterPacket.hpp"
#include "CounterDirectory.hpp"

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

    SendCounterPacket(IBufferWrapper& buffer)
        : m_Buffer(buffer),
          m_ReadyToRead(false)
    {}

    void SendStreamMetaDataPacket() override;

    void SendCounterDirectoryPacket(const ICounterDirectory& counterDirectory) override;

    void SendPeriodicCounterCapturePacket(uint64_t timestamp, const IndexValuePairsVector& values) override;

    void SendPeriodicCounterSelectionPacket(uint32_t capturePeriod,
                                            const std::vector<uint16_t>& selectedCounterIds) override;

    void SetReadyToRead() override;

    static const unsigned int PIPE_MAGIC = 0x45495434;
    static const unsigned int MAX_METADATA_PACKET_LENGTH = 4096;

private:
    template <typename ExceptionType>
    void CancelOperationAndThrow(const std::string& errorMessage)
    {
        // Cancel the operation
        m_Buffer.Commit(0);

        // Throw a runtime exception with the given error message
        throw ExceptionType(errorMessage);
    }

    IBufferWrapper& m_Buffer;
    bool m_ReadyToRead;

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

