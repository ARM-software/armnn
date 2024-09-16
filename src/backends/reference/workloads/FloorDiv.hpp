//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnn
{

template<typename T>
    struct floorDiv
{
    typedef T result_type;
    typedef T first_argument_type;
    T operator () (const T&  inputData0, const T&  inputData1) const
    {
        double result = static_cast<double>(inputData0)/static_cast<double>(inputData1);
        return static_cast<T>(std::floor(result));
    }
};

} //namespace armnn
