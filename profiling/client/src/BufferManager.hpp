//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IBufferManager.hpp"
#include "IConsumer.hpp"

#if !defined(ARMNN_DISABLE_THREADS)
#include <mutex>
#endif

#include <vector>
#include <queue>

namespace arm
{

namespace pipe
{

class BufferManager : public IBufferManager
{
public:
    BufferManager(unsigned int numberOfBuffers = 5, unsigned int maxPacketSize = 4096);

    ~BufferManager() {}

    IPacketBufferPtr Reserve(unsigned int requestedSize, unsigned int& reservedSize) override;

    void Reset();

    void Commit(IPacketBufferPtr& packetBuffer, unsigned int size, bool notifyConsumer = true) override;

    void Release(IPacketBufferPtr& packetBuffer) override;

    IPacketBufferPtr GetReadableBuffer() override;

    void MarkRead(IPacketBufferPtr& packetBuffer) override;

    /// Set Consumer on the buffer manager to be notified when there is a Commit
    /// Can only be one consumer
    void SetConsumer(IConsumer* consumer) override;

    /// Notify the Consumer buffer can be read
    void FlushReadList() override;

private:
    void Initialize();

    // Maximum buffer size
    unsigned int m_MaxBufferSize;
    // Number of buffers
    const unsigned int m_NumberOfBuffers;
    const unsigned int m_MaxNumberOfBuffers;
    unsigned int m_CurrentNumberOfBuffers;

    // List of available packet buffers
    std::vector<IPacketBufferPtr> m_AvailableList;

    // List of readable packet buffers
    std::queue<IPacketBufferPtr> m_ReadableList;

#if !defined(ARMNN_DISABLE_THREADS)
    // Mutex for available packet buffer list
    std::mutex m_AvailableMutex;

    // Mutex for readable packet buffer list
    std::mutex m_ReadableMutex;
#endif

    // Consumer thread to notify packet is ready to read
    IConsumer* m_Consumer = nullptr;
};

} // namespace pipe

} // namespace arm
