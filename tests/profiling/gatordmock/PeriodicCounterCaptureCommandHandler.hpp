//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <common/include/CommandHandlerFunctor.hpp>
#include <common/include/Packet.hpp>

#include <vector>

namespace armnn
{

namespace gatordmock
{

struct CounterCaptureValues
{
    uint64_t m_Timestamp;
    std::vector<uint16_t> m_Uids;
    std::vector<uint32_t> m_Values;
};

class PeriodicCounterCaptureCommandHandler : public arm::pipe::CommandHandlerFunctor
{

public:
    /**
     * @param familyId The family of the packets this handler will service 
     * @param packetId The id of packets this handler will process.
     * @param version The version of that id.
     * @param quietOperation Optional parameter to turn off printouts. This is useful for unittests.
     */
    PeriodicCounterCaptureCommandHandler(uint32_t familyId,
                                         uint32_t packetId,
                                         uint32_t version,
                                         bool quietOperation = false)
        : CommandHandlerFunctor(familyId, packetId, version)
        , m_QuietOperation(quietOperation)
    {}

    void operator()(const arm::pipe::Packet& packet) override;

    CounterCaptureValues m_CounterCaptureValues;

    uint64_t m_CurrentPeriodValue = 0;

private:
    void ParseData(const arm::pipe::Packet& packet);

    uint64_t m_FirstTimestamp = 0, m_SecondTimestamp = 0;

    bool m_HeaderPrinted = false;
    bool m_QuietOperation;
};

}    // namespace gatordmock

}    // namespace armnn
