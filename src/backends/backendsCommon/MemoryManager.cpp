//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "MemoryManager.hpp"

#include <armnn/utility/IgnoreUnused.hpp>

namespace armnn
{

void MemoryManager::StoreMemToAllocate(std::vector<BufferStorage> bufferStorageVector,
                                       std::shared_ptr<ICustomAllocator> customAllocator,
                                       const size_t typeAlignment)
{
    IgnoreUnused(typeAlignment);
    m_AllocatorBufferStoragePairVector.emplace_back(std::make_pair<Allocator, std::vector<BufferStorage>>(
                                                    Allocator{customAllocator},
                                                    std::move(bufferStorageVector)));
}

void MemoryManager::Allocate()
{
    for (auto& m_AllocatorBufferStoragePair : m_AllocatorBufferStoragePairVector)
    {
        auto& allocator = m_AllocatorBufferStoragePair.first;
        for (auto&& bufferStorage : m_AllocatorBufferStoragePair.second)
        {
           bufferStorage.m_Buffer = allocator.m_CustomAllocator->allocate(bufferStorage.m_BufferSize, 0);

            for (auto tensorMemory : bufferStorage.m_TensorMemoryVector)
            {
                tensorMemory->m_Data = allocator.m_CustomAllocator->GetMemoryRegionAtOffset(bufferStorage.m_Buffer,
                                                                                            tensorMemory->m_Offset);
            }
        }
    }
}

void MemoryManager::Deallocate()
{
    for (auto& m_AllocatorBufferStoragePair : m_AllocatorBufferStoragePairVector)
    {
        auto& allocator = m_AllocatorBufferStoragePair.first;
        for (auto&& bufferStorage : m_AllocatorBufferStoragePair.second)
        {
            allocator.m_CustomAllocator->free(bufferStorage.m_Buffer);
        }
    }
}

} // namespace armnn