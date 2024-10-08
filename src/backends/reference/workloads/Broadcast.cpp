//
// Copyright © 2019,2024 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Broadcast.hpp"

namespace armnn
{

BroadcastLoop::BroadcastLoop(const TensorShape& inShape0, const TensorShape& inShape1, const TensorShape& outShape)
: m_DimData(outShape.GetNumDimensions())
{
    const unsigned int numDims = GetNumDimensions();

    unsigned int sIn0 = 1;
    unsigned int sIn1 = 1;
    unsigned int sOut = 1;

    for (unsigned int j = numDims - 1, k = 0; k < numDims ; k++, j--)
    {
        m_DimData[j].m_DimSize = outShape[j];
        m_DimData[j].m_Stride1 = (inShape0[j] > 1) ? sIn0 : 0;
        m_DimData[j].m_Stride2 = (inShape1[j] > 1) ? sIn1 : 0;
        m_DimData[j].m_StrideOut = sOut;

        sIn0 *= inShape0[j];
        sIn1 *= inShape1[j];
        sOut *= outShape[j];
    }
}

BroadcastLoop::BroadcastLoop(const TensorShape& inShape, const TensorShape& outShape)
: m_DimData(outShape.GetNumDimensions())
{
    const unsigned int numDims = GetNumDimensions();

    unsigned int sIn = 1;
    unsigned int sOut = 1;

    // Get the difference between the output dimension and input dimension
    const unsigned int dimDifference = numDims - inShape.GetNumDimensions();

    for (unsigned int j = numDims - 1, k = 0; k < numDims ; k++, j--)
    {

        m_DimData[j].m_DimSize = outShape[j];
        // Pretend there are extra 1-dimensional tensors prepended
        if (dimDifference > 0 && j < dimDifference)
        {
            m_DimData[j].m_Stride1 = 0;
            sIn *= 1;
        }
        else if (dimDifference > 0)
        {
            m_DimData[j].m_Stride1 = (inShape[j - dimDifference] > 1) ? sIn : 0;
            sIn *= inShape[j - dimDifference];
        }
        else
        {
            m_DimData[j].m_Stride1 = (inShape[j] > 1) ? sIn : 0;
            sIn *= inShape[j];
        }
        m_DimData[j].m_StrideOut = sOut;

        sOut *= outShape[j];
    }
}

} // namespace armnn
