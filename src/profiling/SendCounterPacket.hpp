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
    using IndexValuePairsVector = std::vector<std::pair<uint16_t, uint32_t>>;

    SendCounterPacket(IBufferWrapper& buffer) : m_Buffer(buffer), m_ReadyToRead(false) {};

    void SendStreamMetaDataPacket() override;

    void SendCounterDirectoryPacket(const Category& category, const std::vector<Counter>& counters) override;

    void SendPeriodicCounterCapturePacket(uint64_t timestamp, const IndexValuePairsVector& values) override;

    void SendPeriodicCounterSelectionPacket(uint32_t capturePeriod,
                                            const std::vector<uint16_t>& selectedCounterIds) override;

    void SetReadyToRead() override;

    static const unsigned int PIPE_MAGIC = 0x45495434;
    static const unsigned int MAX_METADATA_PACKET_LENGTH = 4096;

private:
    IBufferWrapper& m_Buffer;
    bool m_ReadyToRead;
};

} // namespace profiling

} // namespace armnn

