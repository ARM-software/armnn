//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnn
{

namespace profiling
{

class IReadCounterValue
{
public:
    virtual void GetCounterValue(uint16_t index, uint32_t &value) const = 0;
    virtual ~IReadCounterValue() {}
};

} // namespace profiling

} // namespace armnn