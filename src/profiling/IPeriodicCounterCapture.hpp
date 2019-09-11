//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnn
{

namespace profiling
{
class IPeriodicCounterCapture
{
public:
    virtual void Start() = 0;
    virtual ~IPeriodicCounterCapture() {};
};

} // namespace profiling
} // namespace armnn