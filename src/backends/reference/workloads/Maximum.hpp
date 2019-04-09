//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <iostream>

namespace armnn
{
    template<typename T>
struct maximum : public std::binary_function<T, T, T>
    {
        T
        operator () (const T&  inputData0, const T&  inputData1) const
        {
            return std::max(inputData0, inputData1);
        }
    };

} //namespace armnn
