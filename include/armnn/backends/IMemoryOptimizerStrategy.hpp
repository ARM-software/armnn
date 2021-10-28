//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>

namespace armnn
{

// A MemBlock represents a memory usage requirement in time and space and can be seen as essentially a rectangle
struct MemBlock
{
    MemBlock(const unsigned int startOfLife,
             const unsigned int endOfLife,
             const size_t memSize,
             size_t offset,
             const unsigned int index)
    : m_StartOfLife(startOfLife), m_EndOfLife(endOfLife), m_MemSize(memSize), m_Offset(offset), m_Index(index) {}

    const unsigned int m_StartOfLife; // Y start inclusive
    const unsigned int m_EndOfLife; // Y end inclusive

    const size_t m_MemSize; // Offset + Memsize = X end
    size_t m_Offset; // X start

    const unsigned int m_Index; // Index to keep order
};

// A MemBin represents a single contiguous area of memory that can store 1-n number of MemBlocks
struct MemBin
{
    std::vector<MemBlock> m_MemBlocks;
    size_t m_MemSize;
};

// IMemoryOptimizerStrategy will set m_Offset of the MemBlocks,
// sort them into 1-n bins, then pair each bin of MemBlocks with an int specifying it's total size
// A IMemoryOptimizerStrategy must ensure that
// 1: All MemBlocks have been assigned to a MemBin
// 2: No MemBlock is assigned to multiple MemBins
// 3: No two Memblocks in a MemBin overlap in both the X and Y axis
//    (a strategy cannot change the y axis or length of a MemBlock)
class IMemoryOptimizerStrategy
{
public:
    virtual ~IMemoryOptimizerStrategy() {}

    virtual std::string GetName() const = 0;

    virtual MemBlockStrategyType GetMemBlockStrategyType() const = 0;

    virtual std::vector<MemBin> Optimize(std::vector<MemBlock>& memBlocks) = 0;
};

} // namespace armnn