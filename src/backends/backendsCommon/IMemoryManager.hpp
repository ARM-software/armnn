//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <memory>

namespace armnn
{

class IMemoryManager
{
protected:
    IMemoryManager() {}

public:
    virtual void Acquire() = 0;
    virtual void Release() = 0;

    virtual ~IMemoryManager() {}
};

using IMemoryManagerUniquePtr = std::unique_ptr<IMemoryManager>;

} // namespace armnn