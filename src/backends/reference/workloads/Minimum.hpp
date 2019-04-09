//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnn
{

template<typename T>
struct minimum : public std::binary_function<T, T, T>
{
    T
    operator()(const T& input1, const T& input2) const
    {
        return std::min(input1, input2);
    }
};

} //namespace armnn

