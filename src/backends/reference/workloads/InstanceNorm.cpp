//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "InstanceNorm.hpp"
#include "RefWorkloadUtils.hpp"

#include <armnn/Tensor.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <cmath>

namespace armnn
{

void InstanceNorm(const InstanceNormalizationQueueDescriptor& data,
                  const TensorInfo& inputInfo,
                  Decoder<float>& inputDecoder,
                  Encoder<float>& outputEncoder)
{
    const TensorShape inputShape = inputInfo.GetShape();

    armnnUtils::DataLayoutIndexed dataLayout(data.m_Parameters.m_DataLayout);

    unsigned int inputBatches  = inputShape[0];
    unsigned int inputHeight   = inputShape[dataLayout.GetHeightIndex()];
    unsigned int inputWidth    = inputShape[dataLayout.GetWidthIndex()];
    unsigned int inputChannels = inputShape[dataLayout.GetChannelsIndex()];

    float beta  = data.m_Parameters.m_Beta;
    float eps   = data.m_Parameters.m_Eps;
    float gamma = data.m_Parameters.m_Gamma;

    for (unsigned int n = 0; n < inputBatches; ++n)
    {
        for (unsigned int c = 0; c < inputChannels; ++c)
        {
            float mean = 0, var = 0;

            //Calculate Mean
            for (unsigned int h = 0; h < inputHeight; h++)
            {
                for (unsigned int w = 0; w < inputWidth; w++)
                {
                    unsigned int index = dataLayout.GetIndex(inputShape, n, c, h, w);

                    inputDecoder[index];
                    float value = inputDecoder.Get();
                    mean += value;
                }
            }
            mean /= static_cast<float>(inputHeight * inputWidth);

            //Calculate Variance
            for (unsigned int h = 0; h < inputHeight; h++)
            {
                for (unsigned int w = 0; w < inputWidth; w++)
                {
                    unsigned int index = dataLayout.GetIndex(inputShape, n, c, h, w);

                    inputDecoder[index];
                    float value = inputDecoder.Get();
                    var += (value - mean) * (value - mean);
                }
            }
            var /= static_cast<float>(inputHeight * inputWidth);

            // Apply Instance Normalisation
            for (unsigned int h = 0; h < inputHeight; ++h)
            {
                for (unsigned int w = 0; w < inputWidth; ++w)
                {
                    unsigned int index = dataLayout.GetIndex(inputShape, n, c, h, w);
                    inputDecoder[index];
                    outputEncoder[index];
                    outputEncoder.Set((inputDecoder.Get() - mean) * gamma /  std::sqrt ( var + eps) + beta);
                }

            }
        }
    }
}

} // namespace armnn
