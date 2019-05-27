//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "FullyConnected.hpp"

#include "RefWorkloadUtils.hpp"

#include <boost/assert.hpp>

namespace armnn
{

void FullyConnected(const TensorShape& rInputShape,
                    Decoder<float>& rInputDecoder,
                    const TensorShape& rOutputShape,
                    Encoder<float>& rOutputEncoder,
                    Decoder<float>& rWeightDecoder,
                    Decoder<float>& rBiasDecoder,
                    const bool biasEnabled,
                    const unsigned int K,
                    const bool transposeWeights)
{
    // Perform FullyConnected implementation
    unsigned int outputSize = rOutputShape[1];

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
                    rWeightDecoder[channelOutput * K + channelInput];
                    weight = rWeightDecoder.Get();
                }
                else
                {
                    rWeightDecoder[channelInput * outputSize + channelOutput];
                    weight = rWeightDecoder.Get();
                }

                rInputDecoder[n * K + channelInput];
                outval += weight * rInputDecoder.Get();
            }

            if (biasEnabled)
            {
                rBiasDecoder[channelOutput];
                outval += rBiasDecoder.Get();
            }

            rOutputEncoder[n * outputSize + channelOutput];
            rOutputEncoder.Set(outval);
        }
    }
}

} //namespace armnn
