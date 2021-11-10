//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <unordered_map>
#include <iostream>
#include "StrategyValidator.hpp"

namespace armnn
{

std::vector<MemBin> StrategyValidator::Optimize(std::vector<MemBlock>& memBlocks)
{
    // Condition #1: All Memblocks have been assigned to a MemBin

    // Condition #2: No Memblock is assigned to multiple MemBins

    // Condition #3: No two Memblocks in a MemBin overlap in both the X and Y axis
    //               Memblocks in a MemBin can overlap on the X axis for SingleAxisPacking
    //               Memblocks in a MemBin can overlap on the Y axis or the X for MultiAxisPacking but not both

    std::unordered_map<unsigned int, bool> validationMap;

    for (auto memBlock : memBlocks)
    {
        validationMap[memBlock.m_Index] = false;
    }

    auto memBinVect = m_Strategy->Optimize(memBlocks);

    // Compare each of the input memblocks against every assignedBlock in each bin
    // if we get through all bins without finding a block return
    // if at any stage the block is found twice return

    for (auto memBin : memBinVect)
    {
        for (auto block : memBin.m_MemBlocks)
        {
            try
            {
                if (!validationMap.at(block.m_Index))
                {
                    validationMap.at(block.m_Index) = true;
                }
                else
                {
                    throw MemoryValidationException("Condition #2: Memblock is assigned to multiple MemBins");
                }
            }
            catch (const std::out_of_range&)
            {
                throw MemoryValidationException("Unknown index ");
            }
        }
    }

    for (auto memBlock : memBlocks)
    {
        if (!validationMap.at(memBlock.m_Index))
        {
            throw MemoryValidationException("Condition #1: Block not found in any bin");
        }
    }

    // Check for overlaps once we know blocks are all assigned and no duplicates
    for (auto bin : memBinVect)
    {
        for (unsigned int i = 0; i < bin.m_MemBlocks.size(); ++i)
        {
            auto Block1 = bin.m_MemBlocks[i];
            auto B1Left = Block1.m_Offset;
            auto B1Right = Block1.m_Offset + Block1.m_MemSize;

            auto B1Top = Block1.m_StartOfLife;
            auto B1Bottom = Block1.m_EndOfLife;

            // Only compare with blocks after the current one as previous have already been checked
            for (unsigned int j = i + 1; j < bin.m_MemBlocks.size(); ++j)
            {
                auto Block2 = bin.m_MemBlocks[j];
                auto B2Left = Block2.m_Offset;
                auto B2Right = Block2.m_Offset + Block2.m_MemSize;

                auto B2Top = Block2.m_StartOfLife;
                auto B2Bottom = Block2.m_EndOfLife;

                switch (m_Strategy->GetMemBlockStrategyType())
                {
                    case (MemBlockStrategyType::SingleAxisPacking):
                    {
                        if (B1Top <= B2Bottom && B1Bottom >= B2Top)
                        {
                            throw MemoryValidationException("Condition #3: "
                                        "invalid as two Memblocks overlap on the Y axis for SingleAxisPacking");

                        }
                        break;
                    }
                    case (MemBlockStrategyType::MultiAxisPacking):
                    {
                        // If overlapping on both X and Y then invalid
                        if (B1Left <= B2Right && B1Right >= B2Left &&
                            B1Top <= B2Bottom && B1Bottom >= B2Top)
                        {
                            // Condition #3: two Memblocks overlap on both the X and Y axis
                            throw MemoryValidationException("Condition #3: "
                                                            "two Memblocks overlap on both the X and Y axis");
                        }
                        break;
                    }
                    default:
                        throw MemoryValidationException("Unknown MemBlockStrategyType");
                }
            }
        }
    }

    // None of the conditions broken so return true
    return memBinVect;
}

} // namespace armnn