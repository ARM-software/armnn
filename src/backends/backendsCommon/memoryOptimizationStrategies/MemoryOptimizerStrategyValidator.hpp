//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include <armnn/backends/IMemoryOptimizerStrategy.hpp>

namespace armnn
{

class MemoryOptimizerValidator
{
public:
    explicit MemoryOptimizerValidator(std::shared_ptr<IMemoryOptimizerStrategy> strategy)
        : m_Strategy(strategy)
    {
    };

    bool Validate(std::vector<MemBlock>& memBlocks);

private:
    std::shared_ptr<IMemoryOptimizerStrategy> m_Strategy;
};

} // namespace armnn