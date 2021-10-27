//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include <armnn/backends/IMemoryOptimizerStrategy.hpp>
#include <tuple>
#include <utility>
#include <algorithm>

#include <list>

namespace armnn
{

    /// SingleAxisPriorityList sorts the MemBlocks according to some priority,
    /// then trys to place them into as few bins as possible
    class SingleAxisPriorityList : public IMemoryOptimizerStrategy
    {
    public:
        SingleAxisPriorityList()
                : m_Name(std::string("SingleAxisPriorityList"))
                , m_MemBlockStrategyType(MemBlockStrategyType::SingleAxisPacking) {}

        std::string GetName() const override;

        MemBlockStrategyType GetMemBlockStrategyType() const override;

        std::vector<MemBin> Optimize(std::vector<MemBlock>& memBlocks) override;

    private:

        // Tracks all memBlocks and their positions in a bin as well as their maximum memSize
        struct BinTracker;

        // PlaceBlocks takes a list of MemBlock* and fits them into n bins.
        // A block can only fit into an existing bin if it's lifetime does not overlap with the lifetime of the
        // blocks already in a bin.
        // If no appropriate bin is available a new one is created.
        void PlaceBlocks(const std::list<MemBlock*>& priorityList,
                         std::vector<BinTracker>& placedBlocks,
                         const unsigned int maxLifetime);

        std::string m_Name;
        MemBlockStrategyType m_MemBlockStrategyType;
    };

} // namespace armnn