//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CounterDirectory.hpp"

#include <common/include/CommandHandlerFunctor.hpp>

#include <atomic>

namespace armnn
{

namespace profiling
{

struct CounterDirectoryEventRecord
{
    uint16_t m_CounterClass;
    std::string m_CounterDescription;
    uint16_t m_CounterInterpolation;
    double m_CounterMultiplier;
    std::string m_CounterName;
    uint16_t m_CounterSetUid;
    uint16_t m_CounterUid;
    Optional<std::string> m_CounterUnits;
    uint16_t m_DeviceUid;
    uint16_t m_MaxCounterUid;
};

class DirectoryCaptureCommandHandler : public arm::pipe::CommandHandlerFunctor
{

public:
    DirectoryCaptureCommandHandler(uint32_t familyId, uint32_t packetId, uint32_t version, bool quietOperation = true)
        : CommandHandlerFunctor(familyId, packetId, version)
        , m_QuietOperation(quietOperation)
        , m_AlreadyParsed(false)
    {}

    void operator()(const arm::pipe::Packet& packet) override;

    const ICounterDirectory& GetCounterDirectory() const;

    bool ParsedCounterDirectory()
    {
        return m_AlreadyParsed.load();
    }

    /**
     * Given a Uid that came from a copy of the counter directory translate it to the original.
     *
     * @param copyUid
     * @return the original Uid that the copy maps to.
     */
    uint16_t TranslateUIDCopyToOriginal(uint16_t copyUid)
    {
        return m_UidTranslation[copyUid];
    }

private:
    void ParseData(const arm::pipe::Packet& packet);

    void ReadCategoryRecords(const unsigned char* data, uint32_t offset, std::vector<uint32_t> categoryOffsets);

    std::vector<CounterDirectoryEventRecord>
        ReadEventRecords(const unsigned char* data, uint32_t offset, std::vector<uint32_t> eventRecordsOffsets);

    std::string GetStringNameFromBuffer(const unsigned char* data, uint32_t offset);
    bool IsValidChar(unsigned char c);

    CounterDirectory m_CounterDirectory;
    std::unordered_map<uint16_t, uint16_t> m_UidTranslation;
    bool m_QuietOperation;
    // We can only parse the counter directory once per instance. It won't change anyway as it's static
    // per instance of ArmNN.
    std::atomic<bool> m_AlreadyParsed;
};

}    // namespace profiling

}    // namespace armnn
