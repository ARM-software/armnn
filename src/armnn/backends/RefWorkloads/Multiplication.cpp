//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "Multiplication.hpp"

namespace armnn
{

void Multiplication(const float* in0,
                    const float* in1,
                    unsigned int numElements,
                    float* out)
{
    for (unsigned int i = 0; i < numElements; ++i)
    {
        out[i] = in0[i] * in1[i];
    }
}

} //namespace armnn
