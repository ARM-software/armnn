//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Rsqrt.hpp"

#include <cmath>

namespace armnn
{

void Rsqrt(const float* in,
           float* out,
           const TensorInfo& tensorInfo)
{
    for (size_t i = 0; i < tensorInfo.GetNumElements(); i++)
    {
        out[i] = 1.f / sqrtf(in[i]);
    }
}

} //namespace armnn