//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CounterDirectory.hpp"
#include "GatordMockService.hpp"
#include "MockUtils.hpp"


#include "Packet.hpp"
#include "CommandHandlerFunctor.hpp"
#include "SendCounterPacket.hpp"
#include "IPeriodicCounterCapture.hpp"


#include <vector>
#include <thread>
#include <functional>

namespace armnn
{

namespace gatordmock
{

class DirectoryCaptureCommandHandler : public profiling::CommandHandlerFunctor
{

public:

    DirectoryCaptureCommandHandler(uint32_t familyId,
                                   uint32_t packetId,
                                   uint32_t version,
                                   bool quietOperation = false)
            : CommandHandlerFunctor(familyId, packetId, version)
            , m_QuietOperation(quietOperation)
            , m_CounterDirectoryCount(0)
             {}

    void operator()(const armnn::profiling::Packet &packet) override;

    CounterDirectory GetCounterDirectory() const;
    uint32_t GetCounterDirectoryCount() const;

private:
    void ParseData(const armnn::profiling::Packet &packet);

    std::vector<CategoryRecord> ReadCategoryRecords(const unsigned char *const data,
                                              uint32_t offset,
                                              std::vector<uint32_t> categoryOffsets);

    std::vector<CounterSetRecord> ReadCounterSetRecords(const unsigned char *const data,
                                              uint32_t offset,
                                              std::vector<uint32_t> eventRecordsOffsets);

    std::vector<DeviceRecord> ReadDeviceRecords(const unsigned char *const data,
                                              uint32_t offset,
                                              std::vector<uint32_t> eventRecordsOffsets);

    std::vector<EventRecord> ReadEventRecords(const unsigned char *const data,
                                              uint32_t offset,
                                              std::vector<uint32_t> eventRecordsOffsets);

    CounterDirectory m_CounterDirectory;

    bool m_QuietOperation;
    std::atomic<uint32_t> m_CounterDirectoryCount;
};

} // namespace gatordmock

} // namespace armnn
