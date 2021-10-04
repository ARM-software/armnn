//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/BackendRegistry.hpp>

namespace armnn
{

class MemoryOptimizerStrategyLibrary
{
public:
    MemoryOptimizerStrategyLibrary() = default;

    bool SetMemoryOptimizerStrategy(const BackendId& id, const std::string& strategyName);

};

} // namespace armnn