//
// Copyright © 2017,2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <iostream>
#include <sstream>
#include <cstdint>
#include <armnn/Exceptions.hpp>

namespace armnnUtils
{

void CheckValidSize(std::initializer_list<size_t> validInputCounts,
                    size_t actualValue,
                    const char* validExpr,
                    const char* actualExpr,
                    const armnn::CheckLocation& location);

uint32_t NonNegative(const char* expr,
                     int32_t value,
                     const armnn::CheckLocation& location);

int32_t VerifyInt32(const char* expr,
                    int64_t value,
                    const armnn::CheckLocation& location);

}//armnnUtils

#define CHECKED_INT32(VALUE) armnnUtils::VerifyInt32(#VALUE, VALUE, CHECK_LOCATION())

#define CHECK_VALID_SIZE(ACTUAL, ...) \
armnnUtils::CheckValidSize({__VA_ARGS__}, ACTUAL, #__VA_ARGS__, #ACTUAL, CHECK_LOCATION())

#define CHECKED_NON_NEGATIVE(VALUE) armnnUtils::NonNegative(#VALUE, VALUE, CHECK_LOCATION())
