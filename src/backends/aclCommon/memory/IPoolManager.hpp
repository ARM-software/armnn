//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <arm_compute/runtime/IPoolManager.h>

namespace armnn
{

class IPoolManager : public arm_compute::IPoolManager {
public:
    // Allocates all pools within the pool manager
    virtual void AllocatePools() = 0;

    // Releases all pools within the pool manager
    virtual void ReleasePools() = 0;
};

} // namespace armnn