//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cmath>

namespace armnn
{

template<typename T>
struct squaredDifference
{
    typedef T result_type;
    typedef T first_argument_type;

    T
    operator()(const T& input1, const T& input2) const
    {
        float diff = std::minus<>{}(static_cast<float>(input1),static_cast<float>(input2));
        T squaredDiff = armnn::numeric_cast<T>(std::pow(static_cast<float>(diff), 2));
        return squaredDiff;
    }
};

} //namespace armnn
