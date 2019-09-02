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
    SendCounterPacket(IBufferWrapper& buffer) : m_Buffer(buffer), m_ReadyToRead(false) {}

    void SendStreamMetaDataPacket() override;

    void SendCounterDirectoryPacket(const Category& category, const std::vector<Counter>& counters) override;

    void SendPeriodicCounterCapturePacket(uint64_t timestamp, const std::vector<uint32_t>& counterValues,
                                          const std::vector<uint16_t>& counterUids) override;

    void SendPeriodicCounterSelectionPacket(uint32_t capturePeriod,
                                            const std::vector<uint16_t>& selectedCounterIds) override;

    void SetReadyToRead() override;

private:
    IBufferWrapper& m_Buffer;
    bool m_ReadyToRead;
};

} // namespace profiling

} // namespace armnn

