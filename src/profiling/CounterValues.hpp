//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <cstdint>

namespace armnn
{

namespace profiling
{

class IReadCounterValues
{
public:
    virtual uint16_t GetCounterCount() const = 0;
    virtual void GetCounterValue(uint16_t index, uint32_t& value) const = 0;
    virtual ~IReadCounterValues() {}
};

class IWriteCounterValues : public IReadCounterValues
{
public:
    virtual void SetCounterValue(uint16_t index, uint32_t value) = 0;
    virtual void AddCounterValue(uint16_t index, uint32_t value) = 0;
    virtual void SubtractCounterValue(uint16_t index, uint32_t value) = 0;
    virtual void IncrementCounterValue(uint16_t index) = 0;
    virtual void DecrementCounterValue(uint16_t index) = 0;
    virtual ~IWriteCounterValues() {}
};

} // namespace profiling

} // namespace armnn


