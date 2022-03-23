//
// Copyright © 2022 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace arm
{

namespace pipe
{

struct CounterValue
{
    CounterValue(uint16_t id, uint32_t value) :
        counterId(id), counterValue(value) {}
    uint16_t counterId;
    uint32_t counterValue;
};

}    // namespace pipe

}    // namespace arm
