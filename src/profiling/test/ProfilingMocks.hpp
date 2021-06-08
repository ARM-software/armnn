//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <Holder.hpp>
#include <IProfilingConnectionFactory.hpp>
#include <IProfilingServiceStatus.hpp>
#include <ProfilingService.hpp>
#include <ProfilingUtils.hpp>
#include <SendCounterPacket.hpp>
#include <SendThread.hpp>

#include <armnn/Exceptions.hpp>
#include <armnn/Optional.hpp>
#include <armnn/Conversion.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <common/include/ProfilingGuidGenerator.hpp>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace armnn
{

namespace profiling
{

class MockProfilingConnection : public IProfilingConnection
{
public:
    MockProfilingConnection()
        : m_IsOpen(true)
        , m_WrittenData()
        , m_Packet()
    {}

    enum class PacketType
    {
        StreamMetaData,
        ConnectionAcknowledge,
        CounterDirectory,
        ReqCounterDirectory,
        PeriodicCounterSelection,
        PerJobCounterSelection,
        TimelineMessageDirectory,
        PeriodicCounterCapture,
        ActivateTimelineReporting,
        DeactivateTimelineReporting,
        Unknown
    };

    bool IsOpen() const override
    {
        std::lock_guard<std::mutex> lock(m_Mutex);

        return m_IsOpen;
    }

    void Close() override
    {
        std::lock_guard<std::mutex> lock(m_Mutex);

        m_IsOpen = false;
    }

    bool WritePacket(const unsigned char* buffer, uint32_t length) override
    {
        if (buffer == nullptr || length == 0)
        {
            return false;
        }

        uint32_t header = ReadUint32(buffer, 0);

        uint32_t packetFamily = (header >> 26);
        uint32_t packetId = ((header >> 16) & 1023);

        PacketType packetType;

        switch (packetFamily)
        {
            case 0:
                packetType = packetId < 8 ? PacketType(packetId) : PacketType::Unknown;
                break;
            case 1:
                packetType = packetId == 0 ? PacketType::TimelineMessageDirectory : PacketType::Unknown;
                break;
            case 3:
                packetType = packetId == 0 ? PacketType::PeriodicCounterCapture : PacketType::Unknown;
                break;
            default:
                packetType = PacketType::Unknown;
        }

        std::lock_guard<std::mutex> lock(m_Mutex);

        m_WrittenData.push_back({ packetType, length });
        return true;
    }

    long CheckForPacket(const std::pair<PacketType, uint32_t> packetInfo)
    {
        std::lock_guard<std::mutex> lock(m_Mutex);

        if(packetInfo.second != 0)
        {
            return static_cast<long>(std::count(m_WrittenData.begin(), m_WrittenData.end(), packetInfo));
        }
        else
        {
            return static_cast<long>(std::count_if(m_WrittenData.begin(), m_WrittenData.end(),
            [&packetInfo](const std::pair<PacketType, uint32_t> pair) { return packetInfo.first == pair.first; }));
        }
    }

    bool WritePacket(arm::pipe::Packet&& packet)
    {
        std::lock_guard<std::mutex> lock(m_Mutex);

        m_Packet = std::move(packet);
        return true;
    }

    arm::pipe::Packet ReadPacket(uint32_t timeout) override
    {
        IgnoreUnused(timeout);

        // Simulate a delay in the reading process. The default timeout is way too long.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::lock_guard<std::mutex> lock(m_Mutex);
        return std::move(m_Packet);
    }

    unsigned long GetWrittenDataSize()
    {
        std::lock_guard<std::mutex> lock(m_Mutex);

        return static_cast<unsigned long>(m_WrittenData.size());
    }

    void Clear()
    {
        std::lock_guard<std::mutex> lock(m_Mutex);

        m_WrittenData.clear();
    }

private:
    bool m_IsOpen;
    std::vector<std::pair<PacketType, uint32_t>> m_WrittenData;
    arm::pipe::Packet m_Packet;
    mutable std::mutex m_Mutex;
};

class MockProfilingConnectionFactory : public IProfilingConnectionFactory
{
public:
    IProfilingConnectionPtr GetProfilingConnection(const ExternalProfilingOptions& options) const override
    {
        IgnoreUnused(options);
        return std::make_unique<MockProfilingConnection>();
    }
};

class MockPacketBuffer : public IPacketBuffer
{
public:
    MockPacketBuffer(unsigned int maxSize)
        : m_MaxSize(maxSize)
        , m_Size(0)
        , m_Data(std::make_unique<unsigned char[]>(m_MaxSize))
    {}

    ~MockPacketBuffer() {}

    const unsigned char* GetReadableData() const override { return m_Data.get(); }

    unsigned int GetSize() const override { return m_Size; }

    void MarkRead() override { m_Size = 0; }

    void Commit(unsigned int size) override { m_Size = size; }

    void Release() override { m_Size = 0; }

    unsigned char* GetWritableData() override { return m_Data.get(); }

    void Destroy() override {m_Data.reset(nullptr); m_Size = 0; m_MaxSize =0;}

private:
    unsigned int m_MaxSize;
    unsigned int m_Size;
    std::unique_ptr<unsigned char[]> m_Data;
};

class MockBufferManager : public IBufferManager
{
public:
    MockBufferManager(unsigned int size)
    : m_BufferSize(size),
      m_Buffer(std::make_unique<MockPacketBuffer>(size)) {}

    ~MockBufferManager() {}

    IPacketBufferPtr Reserve(unsigned int requestedSize, unsigned int& reservedSize) override
    {
        if (requestedSize > m_BufferSize)
        {
            reservedSize = m_BufferSize;
        }
        else
        {
            reservedSize = requestedSize;
        }

        return std::move(m_Buffer);
    }

    void Commit(IPacketBufferPtr& packetBuffer, unsigned int size, bool notifyConsumer = true) override
    {
        packetBuffer->Commit(size);
        m_Buffer = std::move(packetBuffer);

        if (notifyConsumer)
        {
            FlushReadList();
        }
    }

    IPacketBufferPtr GetReadableBuffer() override
    {
        return std::move(m_Buffer);
    }

    void Release(IPacketBufferPtr& packetBuffer) override
    {
        packetBuffer->Release();
        m_Buffer = std::move(packetBuffer);
    }

    void MarkRead(IPacketBufferPtr& packetBuffer) override
    {
        packetBuffer->MarkRead();
        m_Buffer = std::move(packetBuffer);
    }

    void SetConsumer(IConsumer* consumer) override
   {
        if (consumer != nullptr)
        {
            m_Consumer = consumer;
        }
   }

    void FlushReadList() override
    {
        // notify consumer that packet is ready to read
        if (m_Consumer != nullptr)
        {
            m_Consumer->SetReadyToRead();
        }
    }

private:
    unsigned int m_BufferSize;
    IPacketBufferPtr m_Buffer;
    IConsumer* m_Consumer = nullptr;
};

class MockStreamCounterBuffer : public IBufferManager
{
public:
    MockStreamCounterBuffer(unsigned int maxBufferSize = 4096)
        : m_MaxBufferSize(maxBufferSize)
        , m_BufferList()
        , m_CommittedSize(0)
        , m_ReadableSize(0)
        , m_ReadSize(0)
    {}
    ~MockStreamCounterBuffer() {}

    IPacketBufferPtr Reserve(unsigned int requestedSize, unsigned int& reservedSize) override
    {
        std::lock_guard<std::mutex> lock(m_Mutex);

        reservedSize = 0;
        if (requestedSize > m_MaxBufferSize)
        {
            throw armnn::InvalidArgumentException("The maximum buffer size that can be requested is [" +
                                                  std::to_string(m_MaxBufferSize) + "] bytes");
        }
        reservedSize = requestedSize;
        return std::make_unique<MockPacketBuffer>(requestedSize);
    }

    void Commit(IPacketBufferPtr& packetBuffer, unsigned int size, bool notifyConsumer = true) override
    {
        std::lock_guard<std::mutex> lock(m_Mutex);

        packetBuffer->Commit(size);
        m_BufferList.push_back(std::move(packetBuffer));
        m_CommittedSize += size;

        if (notifyConsumer)
        {
            FlushReadList();
        }
    }

    void Release(IPacketBufferPtr& packetBuffer) override
    {
        std::lock_guard<std::mutex> lock(m_Mutex);

        packetBuffer->Release();
    }

    IPacketBufferPtr GetReadableBuffer() override
    {
        std::lock_guard<std::mutex> lock(m_Mutex);

        if (m_BufferList.empty())
        {
            return nullptr;
        }
        IPacketBufferPtr buffer = std::move(m_BufferList.back());
        m_BufferList.pop_back();
        m_ReadableSize += buffer->GetSize();
        return buffer;
    }

    void MarkRead(IPacketBufferPtr& packetBuffer) override
    {
        std::lock_guard<std::mutex> lock(m_Mutex);

        m_ReadSize += packetBuffer->GetSize();
        packetBuffer->MarkRead();
    }

    void SetConsumer(IConsumer* consumer) override
    {
        if (consumer != nullptr)
        {
            m_Consumer = consumer;
        }
    }

    void FlushReadList() override
    {
        // notify consumer that packet is ready to read
        if (m_Consumer != nullptr)
        {
            m_Consumer->SetReadyToRead();
        }
    }

    unsigned int GetCommittedSize() const { return m_CommittedSize; }
    unsigned int GetReadableSize()  const { return m_ReadableSize;  }
    unsigned int GetReadSize()      const { return m_ReadSize;      }

private:
    // The maximum buffer size when creating a new buffer
    unsigned int m_MaxBufferSize;

    // A list of buffers
    std::vector<IPacketBufferPtr> m_BufferList;

    // The mutex to synchronize this mock's methods
    std::mutex m_Mutex;

    // The total size of the buffers that has been committed for reading
    unsigned int m_CommittedSize;

    // The total size of the buffers that can be read
    unsigned int m_ReadableSize;

    // The total size of the buffers that has already been read
    unsigned int m_ReadSize;

    // Consumer thread to notify packet is ready to read
    IConsumer* m_Consumer = nullptr;
};

class MockSendCounterPacket : public ISendCounterPacket
{
public:
    MockSendCounterPacket(IBufferManager& sendBuffer) : m_BufferManager(sendBuffer) {}

    void SendStreamMetaDataPacket() override
    {
        std::string message("SendStreamMetaDataPacket");
        unsigned int reserved = 0;
        IPacketBufferPtr buffer = m_BufferManager.Reserve(1024, reserved);
        memcpy(buffer->GetWritableData(), message.c_str(), static_cast<unsigned int>(message.size()) + 1);
        m_BufferManager.Commit(buffer, reserved, false);
    }

    void SendCounterDirectoryPacket(const ICounterDirectory& counterDirectory) override
    {
        IgnoreUnused(counterDirectory);

        std::string message("SendCounterDirectoryPacket");
        unsigned int reserved = 0;
        IPacketBufferPtr buffer = m_BufferManager.Reserve(1024, reserved);
        memcpy(buffer->GetWritableData(), message.c_str(), static_cast<unsigned int>(message.size()) + 1);
        m_BufferManager.Commit(buffer, reserved);
    }

    void SendPeriodicCounterCapturePacket(uint64_t timestamp,
                                          const std::vector<CounterValue>& values) override
    {
        IgnoreUnused(timestamp, values);

        std::string message("SendPeriodicCounterCapturePacket");
        unsigned int reserved = 0;
        IPacketBufferPtr buffer = m_BufferManager.Reserve(1024, reserved);
        memcpy(buffer->GetWritableData(), message.c_str(), static_cast<unsigned int>(message.size()) + 1);
        m_BufferManager.Commit(buffer, reserved);
    }

    void SendPeriodicCounterSelectionPacket(uint32_t capturePeriod,
                                            const std::vector<uint16_t>& selectedCounterIds) override
    {
        IgnoreUnused(capturePeriod, selectedCounterIds);

        std::string message("SendPeriodicCounterSelectionPacket");
        unsigned int reserved = 0;
        IPacketBufferPtr buffer = m_BufferManager.Reserve(1024, reserved);
        memcpy(buffer->GetWritableData(), message.c_str(), static_cast<unsigned int>(message.size()) + 1);
        m_BufferManager.Commit(buffer, reserved);
    }

private:
    IBufferManager& m_BufferManager;
};

class MockCounterDirectory : public ICounterDirectory
{
public:
    MockCounterDirectory() = default;
    ~MockCounterDirectory() = default;

    // Register profiling objects
    const Category* RegisterCategory(const std::string& categoryName)
    {
        // Create the category
        CategoryPtr category = std::make_unique<Category>(categoryName);
        ARMNN_ASSERT(category);

        // Get the raw category pointer
        const Category* categoryPtr = category.get();
        ARMNN_ASSERT(categoryPtr);

        // Register the category
        m_Categories.insert(std::move(category));

        return categoryPtr;
    }

    const Device* RegisterDevice(const std::string& deviceName,
                                 uint16_t cores = 0)
    {
        // Get the device UID
        uint16_t deviceUid = GetNextUid();

        // Create the device
        DevicePtr device = std::make_unique<Device>(deviceUid, deviceName, cores);
        ARMNN_ASSERT(device);

        // Get the raw device pointer
        const Device* devicePtr = device.get();
        ARMNN_ASSERT(devicePtr);

        // Register the device
        m_Devices.insert(std::make_pair(deviceUid, std::move(device)));

        return devicePtr;
    }

    const CounterSet* RegisterCounterSet(
            const std::string& counterSetName,
            uint16_t count = 0)
    {
        // Get the counter set UID
        uint16_t counterSetUid = GetNextUid();

        // Create the counter set
        CounterSetPtr counterSet = std::make_unique<CounterSet>(counterSetUid, counterSetName, count);
        ARMNN_ASSERT(counterSet);

        // Get the raw counter set pointer
        const CounterSet* counterSetPtr = counterSet.get();
        ARMNN_ASSERT(counterSetPtr);

        // Register the counter set
        m_CounterSets.insert(std::make_pair(counterSetUid, std::move(counterSet)));

        return counterSetPtr;
    }

    const Counter* RegisterCounter(const BackendId& backendId,
                                   const uint16_t uid,
                                   const std::string& parentCategoryName,
                                   uint16_t counterClass,
                                   uint16_t interpolation,
                                   double multiplier,
                                   const std::string& name,
                                   const std::string& description,
                                   const armnn::Optional<std::string>& units = armnn::EmptyOptional(),
                                   const armnn::Optional<uint16_t>& numberOfCores = armnn::EmptyOptional(),
                                   const armnn::Optional<uint16_t>& deviceUid = armnn::EmptyOptional(),
                                   const armnn::Optional<uint16_t>& counterSetUid = armnn::EmptyOptional())
    {
        IgnoreUnused(backendId);

        // Get the number of cores from the argument only
        uint16_t deviceCores = numberOfCores.has_value() ? numberOfCores.value() : 0;

        // Get the device UID
        uint16_t deviceUidValue = deviceUid.has_value() ? deviceUid.value() : 0;

        // Get the counter set UID
        uint16_t counterSetUidValue = counterSetUid.has_value() ? counterSetUid.value() : 0;

        // Get the counter UIDs and calculate the max counter UID
        std::vector<uint16_t> counterUids = GetNextCounterUids(uid, deviceCores);
        ARMNN_ASSERT(!counterUids.empty());
        uint16_t maxCounterUid = deviceCores <= 1 ? counterUids.front() : counterUids.back();

        // Get the counter units
        const std::string unitsValue = units.has_value() ? units.value() : "";

        // Create the counter
        CounterPtr counter = std::make_shared<Counter>(armnn::profiling::BACKEND_ID,
                                                       counterUids.front(),
                                                       maxCounterUid,
                                                       counterClass,
                                                       interpolation,
                                                       multiplier,
                                                       name,
                                                       description,
                                                       unitsValue,
                                                       deviceUidValue,
                                                       counterSetUidValue);
        ARMNN_ASSERT(counter);

        // Get the raw counter pointer
        const Counter* counterPtr = counter.get();
        ARMNN_ASSERT(counterPtr);

        // Process multiple counters if necessary
        for (uint16_t counterUid : counterUids)
        {
            // Connect the counter to the parent category
            Category* parentCategory = const_cast<Category*>(GetCategory(parentCategoryName));
            ARMNN_ASSERT(parentCategory);
            parentCategory->m_Counters.push_back(counterUid);

            // Register the counter
            m_Counters.insert(std::make_pair(counterUid, counter));
        }

        return counterPtr;
    }

    // Getters for counts
    uint16_t GetCategoryCount()   const override { return armnn::numeric_cast<uint16_t>(m_Categories.size());  }
    uint16_t GetDeviceCount()     const override { return armnn::numeric_cast<uint16_t>(m_Devices.size());     }
    uint16_t GetCounterSetCount() const override { return armnn::numeric_cast<uint16_t>(m_CounterSets.size()); }
    uint16_t GetCounterCount()    const override { return armnn::numeric_cast<uint16_t>(m_Counters.size());    }

    // Getters for collections
    const Categories&  GetCategories()  const override { return m_Categories;  }
    const Devices&     GetDevices()     const override { return m_Devices;     }
    const CounterSets& GetCounterSets() const override { return m_CounterSets; }
    const Counters&    GetCounters()    const override { return m_Counters;    }

    // Getters for profiling objects
    const Category* GetCategory(const std::string& name) const override
    {
        auto it = std::find_if(m_Categories.begin(), m_Categories.end(), [&name](const CategoryPtr& category)
        {
            ARMNN_ASSERT(category);

            return category->m_Name == name;
        });

        if (it == m_Categories.end())
        {
            return nullptr;
        }

        return it->get();
    }

    const Device* GetDevice(uint16_t uid) const override
    {
        IgnoreUnused(uid);
        return nullptr; // Not used by the unit tests
    }

    const CounterSet* GetCounterSet(uint16_t uid) const override
    {
        IgnoreUnused(uid);
        return nullptr; // Not used by the unit tests
    }

    const Counter* GetCounter(uint16_t uid) const override
    {
        IgnoreUnused(uid);
        return nullptr; // Not used by the unit tests
    }

private:
    Categories  m_Categories;
    Devices     m_Devices;
    CounterSets m_CounterSets;
    Counters    m_Counters;
};

class MockProfilingService : public ProfilingService
{
public:
    MockProfilingService(MockBufferManager& mockBufferManager,
                         bool isProfilingEnabled,
                         const CaptureData& captureData) :
        m_SendCounterPacket(mockBufferManager),
        m_IsProfilingEnabled(isProfilingEnabled),
        m_CaptureData(captureData)
    {}

    /// Return the next random Guid in the sequence
    ProfilingDynamicGuid NextGuid() override
    {
        return m_GuidGenerator.NextGuid();
    }

    /// Create a ProfilingStaticGuid based on a hash of the string
    ProfilingStaticGuid GenerateStaticId(const std::string& str) override
    {
        return m_GuidGenerator.GenerateStaticId(str);
    }

    std::unique_ptr<ISendTimelinePacket> GetSendTimelinePacket() const override
    {
        return nullptr;
    }

    const ICounterMappings& GetCounterMappings() const override
    {
        return m_CounterMapping;
    }

    ISendCounterPacket& GetSendCounterPacket() override
    {
        return m_SendCounterPacket;
    }

    bool IsProfilingEnabled() const override
    {
        return m_IsProfilingEnabled;
    }

    CaptureData GetCaptureData() override
    {
        CaptureData copy(m_CaptureData);
        return copy;
    }

    void RegisterMapping(uint16_t globalCounterId,
                         uint16_t backendCounterId,
                         const armnn::BackendId& backendId)
    {
        m_CounterMapping.RegisterMapping(globalCounterId, backendCounterId, backendId);
    }

    void Reset()
    {
        m_CounterMapping.Reset();
    }

private:
    ProfilingGuidGenerator m_GuidGenerator;
    CounterIdMap           m_CounterMapping;
    SendCounterPacket      m_SendCounterPacket;
    bool                   m_IsProfilingEnabled;
    CaptureData            m_CaptureData;
};

class MockProfilingServiceStatus : public IProfilingServiceStatus
{
public:
    void NotifyProfilingServiceActive() override {}
    void WaitForProfilingServiceActivation(unsigned int timeout) override { IgnoreUnused(timeout); }
};

} // namespace profiling

} // namespace armnn
