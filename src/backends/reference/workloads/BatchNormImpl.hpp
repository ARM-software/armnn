//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefWorkloadUtils.hpp"
#include "TensorBufferArrayView.hpp"

#include <armnn/Tensor.hpp>

#include <DataLayoutIndexed.hpp>

#include <cmath>

namespace armnn
{

template<typename NormData>
static void BatchNormImpl(NormData     data,
                          const float* varIn,
                          const float* meanIn,
                          const float* gammaIn,
                          const float* betaIn,
                          float*       outputData,
                          const float* inputData)
{
    const TensorInfo& inputInfo = GetTensorInfo(data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(data.m_Outputs[0]);

    TensorBufferArrayView<const float> input(inputInfo.GetShape(),
                                             inputData,
                                             data.m_Parameters.m_DataLayout);
    TensorBufferArrayView<float> output(outputInfo.GetShape(),
                                        outputData,
                                        data.m_Parameters.m_DataLayout);

    armnnUtils::DataLayoutIndexed dataLayout(data.m_Parameters.m_DataLayout);

    for (unsigned int c = 0; c < inputInfo.GetShape()[dataLayout.GetChannelsIndex()]; c++)
    {
        float var   = varIn[c];
        float mean  = meanIn[c];
        float gamma = gammaIn[c];
        float beta  = betaIn[c];

        float mult = gamma / sqrtf(var + data.m_Parameters.m_Eps);
        float add  = beta - mult * mean;

        for (unsigned int n = 0; n < inputInfo.GetShape()[0]; n++)
        {
            for (unsigned int h = 0; h < inputInfo.GetShape()[dataLayout.GetHeightIndex()]; h++)
            {
                for (unsigned int w = 0; w < inputInfo.GetShape()[dataLayout.GetWidthIndex()]; w++)
                {
                    output.Get(n, c, h, w) = mult * input.Get(n, c, h, w) + add;
                }
            }
        }
    }
}

} //namespace armnn
