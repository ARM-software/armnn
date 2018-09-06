//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Subtraction.hpp"
#include "Broadcast.hpp"

#include <functional>

namespace
{

void ElementwiseSubtraction(unsigned int numElements, const float* inData0, const float* inData1, float* outData)
{
    for (unsigned int i = 0; i < numElements; ++i)
    {
        outData[i] = inData0[i] - inData1[i];
    }
}

} // namespace

namespace armnn
{

void Subtraction(const TensorShape& inShape0,
                 const TensorShape& inShape1,
                 const TensorShape& outShape,
                 const float* inData0,
                 const float* inData1,
                 float* outData)
{
    if (inShape0 == inShape1)
    {
        ElementwiseSubtraction(inShape0.GetNumElements(), inData0, inData1, outData);
    }
    else
    {
        BroadcastLoop(inShape0, inShape1, outShape).Unroll(std::minus<float>(), 0, inData0, inData1, outData);
    }
}

} //namespace armnn
