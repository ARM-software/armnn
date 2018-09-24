//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefWorkloadUtils.hpp"

#include <armnn/Tensor.hpp>

#include <cmath>

namespace armnn
{

template<typename NormData>
static void BatchNormImpl(NormData data,
                          const float* varIn,
                          const float* meanIn,
                          const float* gammaIn,
                          const float* betaIn,
                          float * outputData,
                          const float * inputData)
{
    const TensorInfo& inputInfo0 = GetTensorInfo(data.m_Inputs[0]);
    for (unsigned int c = 0; c < inputInfo0.GetShape()[1]; c++)
    {
        float var   = varIn[c];
        float mean  = meanIn[c];
        float gamma = gammaIn[c];
        float beta  = betaIn[c];

        float mult = gamma / sqrtf(var + data.m_Parameters.m_Eps);
        float add  = beta - mult * mean;

        for (unsigned int n = 0; n < inputInfo0.GetShape()[0]; n++)
        {
            for (unsigned int j = 0; j < inputInfo0.GetShape()[2]; j++)
            {
                for (unsigned int i = 0; i < inputInfo0.GetShape()[3]; i++)
                {
                    unsigned int index = i +
                                         j*inputInfo0.GetShape()[3] +
                                         c*inputInfo0.GetShape()[3] * inputInfo0.GetShape()[2] +
                                         n*inputInfo0.GetShape()[3] * inputInfo0.GetShape()[2]
                                                                     * inputInfo0.GetShape()[1];

                    outputData[index] = mult * inputData[index] + add;
                }
            }
        }
    }
}

} //namespace armnn
