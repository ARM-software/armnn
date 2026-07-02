//
// Copyright © 2017, 2026 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <type_traits>

// Set style to round to nearest
#ifndef HALF_ROUND_STYLE
    #define HALF_ROUND_STYLE 1
#endif
#ifndef HALF_ROUND_TIES_TO_EVEN
    #define HALF_ROUND_TIES_TO_EVEN 1
#endif

#include "half/half.hpp"

namespace armnn
{
    using Half = half_float::half; //import half float implementation

template<typename T>
struct IsArmnnHalf
    : std::is_same<typename std::remove_cv<T>::type, Half>
{};

template<typename T>
struct IsFloatingPoint
    : std::integral_constant<bool, std::is_floating_point<T>::value || IsArmnnHalf<T>::value>
{};

} //namespace armnn
