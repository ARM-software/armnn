//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/IMemoryOptimizerStrategy.hpp>

#include <algorithm>

namespace armnn
{

class MemoryOptimizerStrategyFactory
{
public:
    MemoryOptimizerStrategyFactory() {}

    template <typename T>
    std::shared_ptr<IMemoryOptimizerStrategy> CreateMemoryOptimizerStrategy()
    {
        return std::make_shared<T>();
    }

};

} // namespace armnn