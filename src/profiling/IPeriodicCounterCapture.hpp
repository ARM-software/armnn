//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace arm
{
namespace pipe
{

class IPeriodicCounterCapture
{
public:
    virtual ~IPeriodicCounterCapture() {}

    virtual void Start() = 0;
    virtual void Stop() = 0;
};

} // namespace pipe
} // namespace arm
