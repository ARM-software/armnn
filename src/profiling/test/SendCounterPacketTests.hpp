//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "../SendCounterPacket.hpp"
#include "../ProfilingUtils.hpp"

#include <armnn/Exceptions.hpp>

#include <boost/test/unit_test.hpp>

#include <chrono>
#include <iostream>

using namespace armnn::profiling;

class MockBuffer : public IBufferWrapper
{
public:
    MockBuffer(unsigned int size)
            : m_BufferSize(size),
              m_Buffer(std::make_unique<unsigned char[]>(size)) {}

    unsigned char* Reserve(unsigned int requestedSize, unsigned int& reservedSize) override
    {
        if (requestedSize > m_BufferSize)
        {
            reservedSize = m_BufferSize;
        }
        else
        {
            reservedSize = requestedSize;
        }

        return m_Buffer.get();
    }

    void Commit(unsigned int size) override {}

    const unsigned char* GetReadBuffer(unsigned int& size) override
    {
        size = static_cast<unsigned int>(strlen(reinterpret_cast<const char*>(m_Buffer.get())) + 1);
        return m_Buffer.get();
    }

    void Release( unsigned int size) override {}

private:
    unsigned int m_BufferSize;
    std::unique_ptr<unsigned char[]> m_Buffer;
};

class MockSendCounterPacket : public ISendCounterPacket
{
public:
    MockSendCounterPacket(IBufferWrapper& sendBuffer) : m_Buffer(sendBuffer) {}

    void SendStreamMetaDataPacket() override
    {
        std::string message("SendStreamMetaDataPacket");
        unsigned int reserved = 0;
        unsigned char* buffer = m_Buffer.Reserve(1024, reserved);
        memcpy(buffer, message.c_str(), static_cast<unsigned int>(message.size()) + 1);
    }

    void SendCounterDirectoryPacket(const ICounterDirectory& counterDirectory) override
    {
        std::string message("SendCounterDirectoryPacket");
        unsigned int reserved = 0;
        unsigned char* buffer = m_Buffer.Reserve(1024, reserved);
        memcpy(buffer, message.c_str(), static_cast<unsigned int>(message.size()) + 1);
    }

    void SendPeriodicCounterCapturePacket(uint64_t timestamp,
                                          const std::vector<std::pair<uint16_t, uint32_t>>& values) override
    {
        std::string message("SendPeriodicCounterCapturePacket");
        unsigned int reserved = 0;
        unsigned char* buffer = m_Buffer.Reserve(1024, reserved);
        memcpy(buffer, message.c_str(), static_cast<unsigned int>(message.size()) + 1);
    }

    void SendPeriodicCounterSelectionPacket(uint32_t capturePeriod,
                                            const std::vector<uint16_t>& selectedCounterIds) override
    {
        std::string message("SendPeriodicCounterSelectionPacket");
        unsigned int reserved = 0;
        unsigned char* buffer = m_Buffer.Reserve(1024, reserved);
        memcpy(buffer, message.c_str(), static_cast<unsigned int>(message.size()) + 1);
        m_Buffer.Commit(reserved);
    }

    void SetReadyToRead() override
    {}

private:
    IBufferWrapper& m_Buffer;
};

class MockCounterDirectory : public ICounterDirectory
{
public:
    MockCounterDirectory() = default;
    ~MockCounterDirectory() = default;

    // Register profiling objects
    const Category* RegisterCategory(const std::string& categoryName,
                                     const armnn::Optional<uint16_t>& deviceUid = armnn::EmptyOptional(),
                                     const armnn::Optional<uint16_t>& counterSetUid = armnn::EmptyOptional())
    {
        // Get the device UID
        uint16_t deviceUidValue = deviceUid.has_value() ? deviceUid.value() : 0;

        // Get the counter set UID
        uint16_t counterSetUidValue = counterSetUid.has_value() ? counterSetUid.value() : 0;

        // Create the category
        CategoryPtr category = std::make_unique<Category>(categoryName, deviceUidValue, counterSetUidValue);
        BOOST_ASSERT(category);

        // Get the raw category pointer
        const Category* categoryPtr = category.get();
        BOOST_ASSERT(categoryPtr);

        // Register the category
        m_Categories.insert(std::move(category));

        return categoryPtr;
    }

    const Device* RegisterDevice(const std::string& deviceName,
                                 uint16_t cores = 0,
                                 const armnn::Optional<std::string>& parentCategoryName = armnn::EmptyOptional())
    {
        // Get the device UID
        uint16_t deviceUid = GetNextUid();

        // Create the device
        DevicePtr device = std::make_unique<Device>(deviceUid, deviceName, cores);
        BOOST_ASSERT(device);

        // Get the raw device pointer
        const Device* devicePtr = device.get();
        BOOST_ASSERT(devicePtr);

        // Register the device
        m_Devices.insert(std::make_pair(deviceUid, std::move(device)));

        // Connect the counter set to the parent category, if required
        if (parentCategoryName.has_value())
        {
            // Set the counter set UID in the parent category
            Category* parentCategory = const_cast<Category*>(GetCategory(parentCategoryName.value()));
            BOOST_ASSERT(parentCategory);
            parentCategory->m_DeviceUid = deviceUid;
        }

        return devicePtr;
    }

    const CounterSet* RegisterCounterSet(
            const std::string& counterSetName,
            uint16_t count = 0,
            const armnn::Optional<std::string>& parentCategoryName = armnn::EmptyOptional())
    {
        // Get the counter set UID
        uint16_t counterSetUid = GetNextUid();

        // Create the counter set
        CounterSetPtr counterSet = std::make_unique<CounterSet>(counterSetUid, counterSetName, count);
        BOOST_ASSERT(counterSet);

        // Get the raw counter set pointer
        const CounterSet* counterSetPtr = counterSet.get();
        BOOST_ASSERT(counterSetPtr);

        // Register the counter set
        m_CounterSets.insert(std::make_pair(counterSetUid, std::move(counterSet)));

        // Connect the counter set to the parent category, if required
        if (parentCategoryName.has_value())
        {
            // Set the counter set UID in the parent category
            Category* parentCategory = const_cast<Category*>(GetCategory(parentCategoryName.value()));
            BOOST_ASSERT(parentCategory);
            parentCategory->m_CounterSetUid = counterSetUid;
        }

        return counterSetPtr;
    }

    const Counter* RegisterCounter(const std::string& parentCategoryName,
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
        // Get the number of cores from the argument only
        uint16_t deviceCores = numberOfCores.has_value() ? numberOfCores.value() : 0;

        // Get the device UID
        uint16_t deviceUidValue = deviceUid.has_value() ? deviceUid.value() : 0;

        // Get the counter set UID
        uint16_t counterSetUidValue = counterSetUid.has_value() ? counterSetUid.value() : 0;

        // Get the counter UIDs and calculate the max counter UID
        std::vector<uint16_t> counterUids = GetNextCounterUids(deviceCores);
        BOOST_ASSERT(!counterUids.empty());
        uint16_t maxCounterUid = deviceCores <= 1 ? counterUids.front() : counterUids.back();

        // Get the counter units
        const std::string unitsValue = units.has_value() ? units.value() : "";

        // Create the counter
        CounterPtr counter = std::make_shared<Counter>(counterUids.front(),
                                                       maxCounterUid,
                                                       counterClass,
                                                       interpolation,
                                                       multiplier,
                                                       name,
                                                       description,
                                                       unitsValue,
                                                       deviceUidValue,
                                                       counterSetUidValue);
        BOOST_ASSERT(counter);

        // Get the raw counter pointer
        const Counter* counterPtr = counter.get();
        BOOST_ASSERT(counterPtr);

        // Process multiple counters if necessary
        for (uint16_t counterUid : counterUids)
        {
            // Connect the counter to the parent category
            Category* parentCategory = const_cast<Category*>(GetCategory(parentCategoryName));
            BOOST_ASSERT(parentCategory);
            parentCategory->m_Counters.push_back(counterUid);

            // Register the counter
            m_Counters.insert(std::make_pair(counterUid, counter));
        }

        return counterPtr;
    }

    // Getters for counts
    uint16_t GetCategoryCount()   const override { return boost::numeric_cast<uint16_t>(m_Categories.size());  }
    uint16_t GetDeviceCount()     const override { return boost::numeric_cast<uint16_t>(m_Devices.size());     }
    uint16_t GetCounterSetCount() const override { return boost::numeric_cast<uint16_t>(m_CounterSets.size()); }
    uint16_t GetCounterCount()    const override { return boost::numeric_cast<uint16_t>(m_Counters.size());    }

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
            BOOST_ASSERT(category);

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
        return nullptr; // Not used by the unit tests
    }

    const CounterSet* GetCounterSet(uint16_t uid) const override
    {
        return nullptr; // Not used by the unit tests
    }

    const Counter* GetCounter(uint16_t uid) const override
    {
        return nullptr; // Not used by the unit tests
    }

private:
    Categories  m_Categories;
    Devices     m_Devices;
    CounterSets m_CounterSets;
    Counters    m_Counters;
};

class SendCounterPacketTest : public SendCounterPacket
{
public:
    SendCounterPacketTest(IBufferWrapper& buffer)
        : SendCounterPacket(buffer)
    {}

    bool CreateDeviceRecordTest(const DevicePtr& device,
                                DeviceRecord& deviceRecord,
                                std::string& errorMessage)
    {
        return CreateDeviceRecord(device, deviceRecord, errorMessage);
    }

    bool CreateCounterSetRecordTest(const CounterSetPtr& counterSet,
                                    CounterSetRecord& counterSetRecord,
                                    std::string& errorMessage)
    {
        return CreateCounterSetRecord(counterSet, counterSetRecord, errorMessage);
    }

    bool CreateEventRecordTest(const CounterPtr& counter,
                               EventRecord& eventRecord,
                               std::string& errorMessage)
    {
        return CreateEventRecord(counter, eventRecord, errorMessage);
    }

    bool CreateCategoryRecordTest(const CategoryPtr& category,
                                  const Counters& counters,
                                  CategoryRecord& categoryRecord,
                                  std::string& errorMessage)
    {
        return CreateCategoryRecord(category, counters, categoryRecord, errorMessage);
    }
};
