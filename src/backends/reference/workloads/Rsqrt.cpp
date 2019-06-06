//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Rsqrt.hpp"

#include <cmath>

namespace armnn
{

void Rsqrt(Decoder<float>& in,
           Encoder<float>& out,
           const TensorInfo& tensorInfo)
{
    for (unsigned int i = 0; i < tensorInfo.GetNumElements(); ++i)
    {
        out[i];
        in[i];
        out.Set(1.f / sqrtf(in.Get()));
    }
}

} //namespace armnn