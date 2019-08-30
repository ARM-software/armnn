//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CounterDirectory.hpp"

namespace armnn
{

namespace profiling
{

class ISendCounterPacket
{
public:
    /// Create and write a StreamMetaDataPacket in the buffer
    virtual void SendStreamMetaDataPacket() = 0;

    /// Create and write a CounterDirectoryPacket from the parameters to the buffer.
    virtual void SendCounterDirectoryPacket(const Category& category, const std::vector<Counter>& counters) = 0;

    /// Create and write a PeriodicCounterCapturePacket from the parameters to the buffer.
    virtual void SendPeriodicCounterCapturePacket(uint64_t timestamp, const std::vector<uint32_t>& counterValues,
                                                  const std::vector<uint16_t>& counterUids) = 0;

    /// Create and write a PeriodicCounterSelectionPacket from the parameters to the buffer.
    virtual void SendPeriodicCounterSelectionPacket(uint32_t capturePeriod,
                                                    const std::vector<uint16_t>& selectedCounterIds) = 0;

    /// Set a "ready to read" flag in the buffer to notify the reading thread to start reading it.
    virtual void SetReadyToRead() = 0;

};

} // namespace profiling

} // namespace armnn

