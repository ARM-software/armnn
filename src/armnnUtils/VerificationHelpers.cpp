//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "VerificationHelpers.hpp"
#include <armnn/Exceptions.hpp>

#include <fmt/format.h>

using namespace armnn;

namespace armnnUtils
{

void CheckValidSize(std::initializer_list<size_t> validInputCounts,
                    size_t actualValue,
                    const char* validExpr,
                    const char* actualExpr,
                    const CheckLocation& location)
{
    bool isValid = std::any_of(validInputCounts.begin(),
                               validInputCounts.end(),
                               [&actualValue](size_t x) { return x == actualValue; } );
    if (!isValid)
    {
        throw ParseException(fmt::format("{} = {} is not valid, not in {{}}. {}",
                                         actualExpr,
                                         actualValue,
                                         validExpr,
                                         location.AsString()));
    }
}

uint32_t NonNegative(const char* expr,
                     int32_t value,
                     const CheckLocation& location)
{
    if (value < 0)
    {
        throw ParseException(fmt::format("'{}' must be non-negative, received: {} at {}",
                                         expr,
                                         value,
                                         location.AsString()));
    }
    else
    {
        return static_cast<uint32_t>(value);
    }
}

int32_t VerifyInt32(const char* expr,
                     int64_t value,
                     const armnn::CheckLocation& location)
{
    if (value < std::numeric_limits<int>::min()  || value > std::numeric_limits<int>::max())
    {
        throw ParseException(fmt::format("'{}' must should fit into a int32 (ArmNN don't support int64),"
                                         " received: {} at {}",
                                         expr,
                                         value,
                                         location.AsString()));
    }
    else
    {
        return static_cast<int32_t>(value);
    }
}

}// armnnUtils
