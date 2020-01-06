//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BaseIterator.hpp"
#include <armnn/Tensor.hpp>

#include <functional>

namespace armnn
{

struct BroadcastLoop
{
    BroadcastLoop(const TensorShape& inShape0, const TensorShape& inShape1, const TensorShape& outShape);

    BroadcastLoop(const TensorShape& inShape, const TensorShape& outShape);

    unsigned int GetNumDimensions()
    {
        return static_cast<unsigned int>(m_DimData.size());
    }

    template <typename Func, typename DecoderOp, typename EncoderOp>
    void Unroll(Func operationFunc,
                unsigned int dimension,
                DecoderOp& inData0,
                DecoderOp& inData1,
                EncoderOp& outData)
    {
        if (dimension >= GetNumDimensions())
        {
            outData.Set(operationFunc(inData0.Get(), inData1.Get()));
            return;
        }

        unsigned int inData0Movement = 0;
        unsigned int inData1Movement = 0;
        unsigned int outDataMovement = 0;

        for (unsigned int i = 0; i < m_DimData[dimension].m_DimSize; i++)
        {
            Unroll(operationFunc, dimension + 1, inData0, inData1, outData);

            inData0 += m_DimData[dimension].m_Stride1;
            inData1 += m_DimData[dimension].m_Stride2;
            outData += m_DimData[dimension].m_StrideOut;

            inData0Movement += m_DimData[dimension].m_Stride1;
            inData1Movement += m_DimData[dimension].m_Stride2;
            outDataMovement += m_DimData[dimension].m_StrideOut;
        }

        // move iterator back to the start
        inData0 -= inData0Movement;
        inData1 -= inData1Movement;
        outData -= outDataMovement;
    }

    template <typename Func, typename DecoderOp, typename EncoderOp>
    void Unroll(Func operationFunc,
                unsigned int dimension,
                DecoderOp& inData,
                EncoderOp& outData)
    {
        if (dimension >= GetNumDimensions())
        {
            outData.Set(operationFunc(inData.Get()));
            return;
        }

        unsigned int inDataMovement = 0;
        unsigned int outDataMovement = 0;

        for (unsigned int i = 0; i < m_DimData[dimension].m_DimSize; i++)
        {
            Unroll(operationFunc, dimension + 1, inData, outData);

            inData += m_DimData[dimension].m_Stride1;
            outData += m_DimData[dimension].m_StrideOut;

            inDataMovement += m_DimData[dimension].m_Stride1;
            outDataMovement += m_DimData[dimension].m_StrideOut;
        }

        // move iterator back to the start
        inData -= inDataMovement;
        outData -= outDataMovement;
    }

private:
    // Struct to hold the dimension data.
    struct BroadcastDimensionData
    {
        unsigned int m_DimSize;
        unsigned int m_StrideOut;
        unsigned int m_Stride1;
        unsigned int m_Stride2;
    };

    std::vector<BroadcastDimensionData> m_DimData;
};

} //namespace armnn