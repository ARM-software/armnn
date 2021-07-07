//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "FullyConnected.hpp"

#include <armnn/utility/Assert.hpp>

#include "RefWorkloadUtils.hpp"

namespace armnn
{

void FullyConnected(const TensorShape& rInputShape,
                    Decoder<float>& rInputDecoder,
                    const TensorShape& rOutputShape,
                    Encoder<float>& rOutputEncoder,
                    const TensorShape& rWeightsShape,
                    Decoder<float>& rWeightDecoder,
                    Decoder<float>* pBiasDecoder,
                    const bool biasEnabled,
                    const unsigned int K,
                    const bool transposeWeights)
{
    // Perform FullyConnected implementation
    unsigned int outputSize = rOutputShape[1];

    const std::vector<float> decodedInputs = rInputDecoder.DecodeTensor(rInputShape);
    const std::vector<float> decodedWeights = rWeightDecoder.DecodeTensor(rWeightsShape);

    const TensorShape biasShape{outputSize};

    ARMNN_ASSERT(!biasEnabled || pBiasDecoder != nullptr);
    const std::vector<float> decodedBiases = biasEnabled ? pBiasDecoder->DecodeTensor(biasShape) : std::vector<float>();


    for (unsigned int n = 0; n < rInputShape[0]; n++)
    {
        for (unsigned int channelOutput = 0; channelOutput < outputSize; channelOutput++)
        {
            float outval = 0.f;

            for (unsigned int channelInput = 0; channelInput < K; channelInput++)
            {
                float weight;
                if (transposeWeights)
                {
                    weight = decodedWeights[channelOutput * K + channelInput];
                }
                else
                {
                    weight = decodedWeights[channelInput * outputSize + channelOutput];
                }

                outval += weight * decodedInputs[n * K + channelInput];
            }

            if (biasEnabled)
            {
                outval += decodedBiases[channelOutput];
            }

            rOutputEncoder[n * outputSize + channelOutput];
            rOutputEncoder.Set(outval);
        }
    }
}

} //namespace armnn
