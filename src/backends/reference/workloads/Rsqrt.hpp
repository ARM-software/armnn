//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <iostream>

namespace armnn
{
template<typename T>
struct rsqrt
    {
        typedef T result_type;
        typedef T argument_type;

        T
        operator () (const T&  inputData) const
        {
            return 1 / std::sqrt(inputData);
        }
    };

} //namespace armnn
