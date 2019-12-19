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

    IPacketBufferPtr Reserve(unsigned int requestedSize, unsigned int& reservedSize) override;

    void Reset();

    void Commit(IPacketBufferPtr& packetBuffer, unsigned int size) override;

    void Release(IPacketBufferPtr& packetBuffer) override;

    IPacketBufferPtr GetReadableBuffer() override;

    void MarkRead(IPacketBufferPtr& packetBuffer) override;

private:
    void Initialize();

    // Maximum buffer size
    unsigned int m_MaxBufferSize;
    // Number of buffers
    unsigned int m_NumberOfBuffers;

    // List of available packet buffers
    std::vector<IPacketBufferPtr> m_AvailableList;

    // List of readable packet buffers
    std::vector<IPacketBufferPtr> m_ReadableList;

    // Mutex for available packet buffer list
    std::mutex m_AvailableMutex;

    // Mutex for readable packet buffer list
    std::mutex m_ReadableMutex;
};

} // namespace profiling

} // namespace armnn
