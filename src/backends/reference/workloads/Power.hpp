//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <iostream>

namespace armnn
{

template<typename T>
struct power
{
    typedef T result_type;
    typedef T first_argument_type;

    T
    operator()(const T& input1, const T& input2) const
    {
        T power = armnn::numeric_cast<T>(std::pow(static_cast<float>(input1), static_cast<float>(input2)));
        return power;
    }
};

} //namespace armnn
