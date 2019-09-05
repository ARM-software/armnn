//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Abs.hpp"

namespace armnn
{

void Abs(Decoder<float>& in,
         Encoder<float>& out,
         const TensorInfo& tensorInfo)
{
    for (unsigned int i = 0u; i < tensorInfo.GetNumElements(); ++i)
    {
        out[i];
        in[i];
        out.Set(std::abs(in.Get()));
    }
}

} //namespace armnn
