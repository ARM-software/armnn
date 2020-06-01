//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BatchNormImpl.hpp"
#include "RefWorkloadUtils.hpp"

#include <armnn/Tensor.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <cmath>

namespace armnn
{

void BatchNormImpl(const BatchNormalizationQueueDescriptor& data,
                   Decoder<float>& meanDecoder,
                   Decoder<float>& varianceDecoder,
                   Decoder<float>& betaDecoder,
                   Decoder<float>& gammaDecoder,
                   Decoder<float>& inputDecoder,
                   Encoder<float>& outputEncoder)
{
    const TensorInfo& inputInfo = GetTensorInfo(data.m_Inputs[0]);
    const TensorShape inputShape = inputInfo.GetShape();

    armnnUtils::DataLayoutIndexed dataLayout(data.m_Parameters.m_DataLayout);

    unsigned int inputBatches  = inputShape[0];
    unsigned int inputHeight   = inputShape[dataLayout.GetHeightIndex()];
    unsigned int inputWidth    = inputShape[dataLayout.GetWidthIndex()];
    unsigned int inputChannels = inputShape[dataLayout.GetChannelsIndex()];

    for (unsigned int c = 0; c < inputChannels; c++)
    {
        meanDecoder[c];
        varianceDecoder[c];
        betaDecoder[c];
        gammaDecoder[c];
        float mean  = meanDecoder.Get();
        float var   = varianceDecoder.Get();
        float beta  = betaDecoder.Get();
        float gamma = gammaDecoder.Get();

        float mult = gamma / sqrtf(var + data.m_Parameters.m_Eps);
        float add  = beta - mult * mean;

        for (unsigned int n = 0; n < inputBatches; n++)
        {
            for (unsigned int h = 0; h < inputHeight; h++)
            {
                for (unsigned int w = 0; w < inputWidth; w++)
                {
                    unsigned int index = dataLayout.GetIndex(inputShape, n, c, h, w);
                    inputDecoder[index];
                    outputEncoder[index];
                    outputEncoder.Set(mult * inputDecoder.Get() + add);
                }
            }
        }
    }
}

} // namespace armnn
