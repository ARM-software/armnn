//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IBufferManager.hpp"
#include "ProfilingUtils.hpp"

#include <client/include/ISendCounterPacket.hpp>

#include <type_traits>

namespace arm
{

namespace pipe
{

class SendCounterPacket : public ISendCounterPacket
{
public:
    using CategoryRecord        = std::vector<uint32_t>;
    using DeviceRecord          = std::vector<uint32_t>;
    using CounterSetRecord      = std::vector<uint32_t>;
    using EventRecord           = std::vector<uint32_t>;
    using IndexValuePairsVector = std::vector<CounterValue>;

    SendCounterPacket(IBufferManager& buffer,
                      const std::string& softwareInfo,
                      const std::string& softwareVersion,
                      const std::string& hardwareVersion)
        : m_BufferManager(buffer),
          m_SoftwareInfo(softwareInfo),
          m_SoftwareVersion(softwareVersion),
          m_HardwareVersion(hardwareVersion)
    {}

    void SendStreamMetaDataPacket() override;

    void SendCounterDirectoryPacket(const ICounterDirectory& counterDirectory) override;

    void SendPeriodicCounterCapturePacket(uint64_t timestamp, const IndexValuePairsVector& values) override;

    void SendPeriodicCounterSelectionPacket(uint32_t capturePeriod,
                                            const std::vector<uint16_t>& selectedCounterIds) override;

private:
    template <typename ExceptionType>
    void CancelOperationAndThrow(const std::string& errorMessage)
    {
        // Throw a runtime exception with the given error message
        throw ExceptionType(errorMessage);
    }

    template <typename ExceptionType>
    void CancelOperationAndThrow(IPacketBufferPtr& writerBuffer, const std::string& errorMessage)
    {
        if (std::is_same<ExceptionType, arm::pipe::BufferExhaustion>::value)
        {
            m_BufferManager.FlushReadList();
        }

        if (writerBuffer != nullptr)
        {
            // Cancel the operation
            m_BufferManager.Release(writerBuffer);
        }

        // Throw a runtime exception with the given error message
        throw ExceptionType(errorMessage);
    }

    IBufferManager& m_BufferManager;

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
private:
    std::string m_SoftwareInfo;
    std::string m_SoftwareVersion;
    std::string m_HardwareVersion;
};

} // namespace pipe

} // namespace arm
