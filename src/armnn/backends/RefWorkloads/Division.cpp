//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "Division.hpp"
#include "Broadcast.hpp"

#include <functional>

namespace
{

void ElementwiseDivision(unsigned int numElements,
                         const float* inData0,
                         const float* inData1,
                         float* outData)
{
    for (unsigned int i = 0; i < numElements; ++i)
    {
        //TODO How to handle divide by 0
        outData[i] = inData0[i] / inData1[i];
    }
}

} // namespace

namespace armnn
{

void Division(const TensorShape& inShape0,
              const TensorShape& inShape1,
              const TensorShape& outShape,
              const float* inData0,
              const float* inData1,
              float* outData)
{
    if (inShape0 == inShape1)
    {
        ElementwiseDivision(inShape0.GetNumElements(), inData0, inData1, outData);
    }
    else
    {
        BroadcastLoop(inShape0, inShape1, outShape).Unroll(std::divides<float>(),
                                                           0,
                                                           inData0,
                                                           inData1,
                                                           outData);
    }
}

} //namespace armnn
