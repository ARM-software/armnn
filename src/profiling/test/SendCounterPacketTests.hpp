//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <SendCounterPacket.hpp>
#include <SendThread.hpp>
#include <ProfilingUtils.hpp>
#include <IProfilingConnectionFactory.hpp>

#include <common/include/IgnoreUnused.hpp>
#include <common/include/NumericCast.hpp>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace arm
{

namespace pipe
{

class SendCounterPacketTest : public SendCounterPacket
{
public:
    SendCounterPacketTest(IBufferManager& buffer)
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

} // namespace pipe

} // namespace arm
