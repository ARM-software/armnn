//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "Division.hpp"
#include "Broadcast.hpp"

#include <functional>

#include <cmath>

namespace
{

void ElementwiseDivision(unsigned int numElements,
                         const float* inData0,
                         const float* inData1,
                         float* outData)
{
    for (unsigned int i = 0; i < numElements; ++i)
    {
        if (inData1[i] != 0.0f)
        {
            outData[i] = inData0[i] / inData1[i];
        }
        else if (inData0[i] == 0.0f)
        {
            if (!std::signbit(inData1[i]))
            {
                outData[i]= NAN;
            }
            else
            {
                outData[i]= -NAN;
            }
        }
        else if (inData0[i] < 0.0f)
        {
            if (!std::signbit(inData1[i]))
            {
                outData[i] = -INFINITY;
            }
            else
            {
                outData[i] = INFINITY;
            }
        }
        else
        {
            if (!std::signbit(inData1[i]))
            {
                outData[i] = INFINITY;
            }
            else
            {
                outData[i] = -INFINITY;
            }
        }
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
