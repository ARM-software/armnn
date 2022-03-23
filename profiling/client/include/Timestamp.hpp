//
// Copyright © 2022 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CounterValue.hpp"

namespace arm
{

namespace pipe
{

struct Timestamp
{
    uint64_t timestamp;
    std::vector<CounterValue> counterValues;
};

}    // namespace pipe

}    // namespace arm
