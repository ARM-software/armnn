//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SingleAxisPriorityList.hpp"

#include <algorithm>
#include <cstdlib>

#include <iostream>

namespace armnn
{

// This strategy uses a vector of size_ts/words to represent occupancy of a memBlock in a memBin.
// Where each bit represents occupancy of a single time-step in that lifetime.
// We can then use bit masks to check for overlaps of memBlocks along the lifetime

// For more information on the algorithm itself see: https://arxiv.org/pdf/2001.03288.pdf
// This strategy is an implementation of 4.3 Greedy by Size
constexpr size_t wordSize = sizeof(size_t) * 8;

std::string SingleAxisPriorityList::GetName() const {
    return m_Name;
}

MemBlockStrategyType SingleAxisPriorityList::GetMemBlockStrategyType() const {
    return m_MemBlockStrategyType;
}

struct SingleAxisPriorityList::BinTracker
{
    // maxLifeTime is the number of operators in the model
    // We then divide that by wordSize to get the number of words we need to store all the lifetimes
    BinTracker(unsigned int maxLifeTime)
            : m_OccupiedSpaces(1 + maxLifeTime/wordSize, 0)
    {}

    // Add a block of a single word size to the bin
    void AddBlock(MemBlock* block, const size_t word, const size_t index)
    {
        m_OccupiedSpaces[index] = m_OccupiedSpaces[index] | word;

        m_PlacedBlocks.push_back(block);
        m_MemSize = std::max(m_MemSize, block->m_MemSize);
    }

    // Add a block with a word size of two or more to the bin
    void AddBlock(MemBlock* block,
                  const size_t startIndex,
                  const size_t endIndex,
                  const size_t firstWord,
                  const size_t lastWord)
    {
        m_OccupiedSpaces[startIndex] = m_OccupiedSpaces[startIndex] | firstWord;
        m_OccupiedSpaces[endIndex] = m_OccupiedSpaces[endIndex] | lastWord;

        for (size_t i = startIndex +1; i <= endIndex -1; ++i)
        {
            m_OccupiedSpaces[i] = std::numeric_limits<size_t>::max();
        }

        m_PlacedBlocks.push_back(block);
        m_MemSize = std::max(m_MemSize, block->m_MemSize);
    }

    size_t m_MemSize = 0;
    std::vector<size_t> m_OccupiedSpaces;
    std::vector<MemBlock*> m_PlacedBlocks;
};

void SingleAxisPriorityList::PlaceBlocks(const std::list<MemBlock*>& priorityList,
                                         std::vector<BinTracker>& placedBlocks,
                                         const unsigned int maxLifetime)
{
    // This function is used when the given block start and end lifetimes fit within a single word.
    auto singleWordLoop = [&](MemBlock* curBlock, const size_t firstWord, const size_t index)
    {
        bool placed = false;
        // loop through all existing bins
        for (auto& blockList : placedBlocks)
        {
            // Check if the lifetimes at the given index overlap with the lifetimes of the block
            if ((blockList.m_OccupiedSpaces[index] & firstWord) == 0)
            {
                // If the binary AND is 0 there is no overlap between the words and the block will fit
                blockList.AddBlock(curBlock, firstWord, index);
                placed = true;
                break;
            }
        }
        // No suitable bin was found, create a new bin/BinTracker and add it to the placedBlocks vector
        if (!placed)
        {
            placedBlocks.emplace_back(BinTracker{maxLifetime});
            placedBlocks.back().AddBlock(curBlock, firstWord, index);
        }
    };

    // This function is used when the given block start and end lifetimes fit within two words.
    auto doubleWordLoop =[&](MemBlock* curBlock,
                             const size_t firstWord,
                             const size_t firstIndex,
                             const size_t lastWord,
                             const size_t lastIndex)
    {
        bool placed = false;
        for (auto& blockList : placedBlocks)
        {
            // Check if the lifetimes at the given indexes overlap with the lifetimes of the block
            if ((blockList.m_OccupiedSpaces[firstIndex] & firstWord) == 0 &&
                (blockList.m_OccupiedSpaces[lastIndex] & lastWord) == 0)
            {
                blockList.AddBlock(curBlock, firstIndex, lastIndex, firstWord, lastWord);
                placed = true;
                break;
            }
        }
        // No suitable bin was found, create a new bin/BinTracker and add it to the placedBlocks vector
        if (!placed)
        {
            placedBlocks.emplace_back(BinTracker{maxLifetime});
            placedBlocks.back().AddBlock(curBlock, firstIndex, lastIndex, firstWord, lastWord);
        }
    };

    // Loop through the blocks to place
    for(const auto curBlock : priorityList)
    {
        // The lifetimes of both the block and bin are represented by single bits on a word/s
        // Each bin has maxLifetime/wordSize words
        // The number of words for a block depends on the size of the blocks lifetime
        // and the alignment of the block's lifetimes
        // This allows for checking sizeof(size_t) * 8 lifetimes at once with a binary AND
        const size_t firstWordIndex = curBlock->m_StartOfLife/wordSize;
        const size_t lastWordIndex = curBlock->m_EndOfLife/wordSize;

        // Align and right shift the first word
        // This sets the bits before curBlock->m_StartOfLife to 0
        size_t remainder = (curBlock->m_StartOfLife - firstWordIndex * wordSize);
        size_t firstWord = std::numeric_limits<size_t>::max() >> remainder;

        // If the indexes match the block can fit into a single word
        if(firstWordIndex == lastWordIndex)
        {
            // We then need to zero the bits to the right of curBlock->m_EndOfLife
            remainder += curBlock->m_EndOfLife + 1 - curBlock->m_StartOfLife;
            firstWord = firstWord >> (wordSize - remainder);
            firstWord = firstWord << (wordSize - remainder);

            singleWordLoop(curBlock, firstWord, firstWordIndex);
            continue;
        }

        // The indexes don't match we need at least two words
        // Zero the bits to the right of curBlock->m_EndOfLife
        remainder = (curBlock->m_EndOfLife + 1 - lastWordIndex * wordSize);
        size_t lastWord = std::numeric_limits<size_t>::max() << (wordSize - remainder);

        if(firstWordIndex + 1 == lastWordIndex)
        {
            doubleWordLoop(curBlock, firstWord, firstWordIndex, lastWord, lastWordIndex);
            continue;
        }

        // The block cannot fit into two words
        // We don't need to create any more words to represent this,
        // as any word between the first and last block would always equal the maximum value of size_t,
        // all the lifetimes would be occupied

        // Instead, we just check that the corresponding word in the bin is completely empty

        bool placed = false;
        for (auto& blockList : placedBlocks)
        {
            // Check the first and last word
            if ((blockList.m_OccupiedSpaces[firstWordIndex] & firstWord) != 0 ||
                (blockList.m_OccupiedSpaces[lastWordIndex] & lastWord) != 0)
            {
                continue;
            }

            bool fits = true;
            // Check that all spaces in between are clear
            for (size_t i = firstWordIndex +1; i <= lastWordIndex -1; ++i)
            {
                if (blockList.m_OccupiedSpaces[i] != 0)
                {
                    fits = false;
                    break;
                }
            }

            if (!fits)
            {
                continue;
            }

            blockList.AddBlock(curBlock, firstWordIndex, lastWordIndex, firstWord, lastWord);
            placed = true;
            break;
        }

        // No suitable bin was found, create a new bin/BinTracker and add it to the placedBlocks vector
        if (!placed)
        {
            placedBlocks.emplace_back(BinTracker{maxLifetime});
            placedBlocks.back().AddBlock(curBlock, firstWordIndex, lastWordIndex, firstWord, lastWord);
        }
    }
}

std::vector<MemBin> SingleAxisPriorityList::Optimize(std::vector<MemBlock>& blocks)
{
    unsigned int maxLifetime = 0;
    std::list<MemBlock*> priorityList;
    for (auto& block: blocks)
    {
        maxLifetime = std::max(maxLifetime, block.m_EndOfLife);
        priorityList.emplace_back(&block);
    }
    maxLifetime++;

    // From testing ordering by m_MemSize in non-descending order gives the best results overall
    priorityList.sort([](const MemBlock* lhs, const MemBlock* rhs)
                      {
                          return  lhs->m_MemSize > rhs->m_MemSize ;
                      });


    std::vector<BinTracker> placedBlocks;
    placedBlocks.reserve(maxLifetime);
    PlaceBlocks(priorityList, placedBlocks, maxLifetime);

    std::vector<MemBin> bins;
    bins.reserve(placedBlocks.size());
    for (auto blockList: placedBlocks)
    {
        MemBin bin;
        bin.m_MemBlocks.reserve(blockList.m_PlacedBlocks.size());
        bin.m_MemSize = blockList.m_MemSize;

        for (auto block : blockList.m_PlacedBlocks)
        {
            bin.m_MemBlocks.emplace_back(MemBlock{block->m_StartOfLife,
                                                  block->m_EndOfLife,
                                                  block->m_MemSize,
                                                  0,
                                                  block->m_Index,});
        }
        bins.push_back(std::move(bin));
    }

    return bins;
}

} // namespace armnn

