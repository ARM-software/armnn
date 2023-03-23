//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <iostream>

namespace armnn
{
template<typename T>
struct ceil
    {
        typedef T result_type;
        typedef T argument_type;

        T
        operator () (const T&  inputData) const
        {
            return std::ceil(inputData);
        }
    };

} //namespace armnn
