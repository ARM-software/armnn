//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Fill.hpp"

#include "RefWorkloadUtils.hpp"

namespace armnn
{

void Fill(Encoder<float>& output,
          const TensorShape& desiredOutputShape,
          const float value)
{
    for(unsigned int i = 0; i < desiredOutputShape.GetNumElements(); ++i)
    {
        output[i];
        output.Set(value);
    }
}

} //namespace armnn
