//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstdint>

namespace arm
{

namespace pipe
{

class IReadCounterValues
{
public:
    virtual ~IReadCounterValues() {}

    virtual bool IsCounterRegistered(uint16_t counterUid) const = 0;
    virtual bool IsCounterRegistered(const std::string& counterName) const = 0;
    virtual uint16_t GetCounterCount() const = 0;
    virtual uint32_t GetAbsoluteCounterValue(uint16_t counterUid) const = 0;
    virtual uint32_t GetDeltaCounterValue(uint16_t counterUid) = 0;
};

class IWriteCounterValues
{
public:
    virtual ~IWriteCounterValues() {}

    virtual void SetCounterValue(uint16_t counterUid, uint32_t value) = 0;
    virtual uint32_t AddCounterValue(uint16_t counterUid, uint32_t value) = 0;
    virtual uint32_t SubtractCounterValue(uint16_t counterUid, uint32_t value) = 0;
    virtual uint32_t IncrementCounterValue(uint16_t counterUid) = 0;
};

class IReadWriteCounterValues : public IReadCounterValues, public IWriteCounterValues
{
public:
    virtual ~IReadWriteCounterValues() {}
};

} // namespace pipe

} // namespace arm
