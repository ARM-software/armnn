//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IBufferManager.hpp"

#include <condition_variable>
#include <mutex>
#include <vector>

namespace armnn
{

namespace profiling
{

class BufferManager : public IBufferManager
{
public:
    BufferManager(unsigned int numberOfBuffers = 5, unsigned int maxPacketSize = 4096);

    ~BufferManager() {}

    std::unique_ptr<IPacketBuffer> Reserve(unsigned int requestedSize, unsigned int& reservedSize) override;

    void Commit(std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int size) override;

    void Release(std::unique_ptr<IPacketBuffer>& packetBuffer) override;

    std::unique_ptr<IPacketBuffer> GetReadableBuffer() override;

    void MarkRead(std::unique_ptr<IPacketBuffer>& packetBuffer) override;

private:
    // Maximum buffer size
    unsigned int m_MaxBufferSize;

    // List of available packet buffers
    std::vector<std::unique_ptr<IPacketBuffer>> m_AvailableList;

    // List of readable packet buffers
    std::vector<std::unique_ptr<IPacketBuffer>> m_ReadableList;

    // Mutex for available packet buffer list
    std::mutex m_AvailableMutex;

    // Mutex for readable packet buffer list
    std::mutex m_ReadableMutex;

    // Condition to notify when data is availabe to be read
    std::condition_variable m_ReadDataAvailable;
};

} // namespace profiling

} // namespace armnn
