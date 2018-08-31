//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include <armnn/Tensor.hpp>

#include <functional>

namespace armnn
{

struct BroadcastLoop
{
    BroadcastLoop(const TensorShape& inShape0, const TensorShape& inShape1, const TensorShape& outShape);

    unsigned int GetNumDimensions()
    {
        return static_cast<unsigned int>(m_DimData.size());
    }

    template <typename T0, typename T1, typename U, typename Func>
    void Unroll(Func operationFunc,
                unsigned int dimension,
                const T0* inData0,
                const T1* inData1,
                U* outData)
    {
        if (dimension >= GetNumDimensions())
        {
            *outData = operationFunc(*inData0, *inData1);
            return;
        }

        for (unsigned int i = 0; i < m_DimData[dimension].m_DimSize; i++)
        {
            Unroll(operationFunc, dimension + 1, inData0, inData1, outData);

            inData0 += m_DimData[dimension].m_Stride1;
            inData1 += m_DimData[dimension].m_Stride2;
            outData += m_DimData[dimension].m_StrideOut;
        }
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