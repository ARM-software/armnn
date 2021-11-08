//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/IMemoryOptimizerStrategy.hpp>

#include <algorithm>

namespace armnn
{

struct IMemoryOptimizerStrategyFactory
{
    virtual ~IMemoryOptimizerStrategyFactory() = default;
    virtual std::unique_ptr<IMemoryOptimizerStrategy> CreateMemoryOptimizerStrategy() = 0;
};

template <typename T>
struct StrategyFactory : public IMemoryOptimizerStrategyFactory
{
    std::unique_ptr<IMemoryOptimizerStrategy> CreateMemoryOptimizerStrategy() override
    {
        return std::make_unique<T>();
    }
};

} // namespace armnn