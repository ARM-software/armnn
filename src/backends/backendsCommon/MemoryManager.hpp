//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/ICustomAllocator.hpp>

namespace armnn
{
struct Allocator
{
    /// Pointer to @ICustomAllocator.
    std::shared_ptr<ICustomAllocator> m_CustomAllocator{};
    /// Value which the size of each buffer (actual data size + padding) has to be a multiple of.
    size_t m_Alignment = 0 ;
};

struct TensorMemory
{
    /// Number of bytes the value is away from the @BufferStorage.m_Buffer.
    size_t m_Offset{};
    /// Identifier to be used by the @LoadedNetwork to order the tensors.
    unsigned int m_OutputSlotId{};
    /// Pointer to the tensor value.
    void* m_Data = nullptr;
};

struct BufferStorage
{
    /// Vector of pointer to @TensorMemory.
    std::vector<std::shared_ptr<TensorMemory>> m_TensorMemoryVector;
    /// Total size of the buffer.
    size_t m_BufferSize;
    /// Pointer to the first element of the buffer.
    void* m_Buffer = nullptr;
};

class MemoryManager
{
public:
    /// Initialization method to store in m_AllocatorBufferStoragePairVector all information needed.
    /// @param[in] bufferStorageVector - Vector of BufferStorage.
    /// @param[in] customAllocator - Pointer to ICustomAllocator.
    /// @param[in] typeAlignment - Optional parameter. Value of which the size of each value has to be multiple of.
    void StoreMemToAllocate(std::vector<BufferStorage> bufferStorageVector,
                            std::shared_ptr<ICustomAllocator> customAllocator,
                            size_t typeAlignment = 0);

    /// Allocate the amount of memory indicated by m_BufferSize, and
    /// point each m_Data to each correspondent Tensor so that they are m_Offset bytes separated.
    void Allocate();

    /// Deallocate memory
    void Deallocate();

private:
    std::vector<std::pair<Allocator, std::vector<BufferStorage>>> m_AllocatorBufferStoragePairVector;
};

} // namespace armnn
