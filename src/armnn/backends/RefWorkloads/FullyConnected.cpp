//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "FullyConnected.hpp"

#include <boost/assert.hpp>

namespace armnn
{

void FullyConnected(const float*      inputData,
                    float*            outputData,
                    const TensorInfo& inputTensorInfo,
                    const TensorInfo& outputTensorInfo,
                    const float*      weightData,
                    const float*      biasData,
                    bool              transposeWeights)
{
    unsigned int N = outputTensorInfo.GetShape()[1]; // Outputs Vector Size.

    BOOST_ASSERT(inputTensorInfo.GetNumDimensions() > 1); // Needs some data.

    unsigned int K = 1; // Total number of activations in the input.
    for (unsigned int i = 1; i < inputTensorInfo.GetNumDimensions(); i++)
    {
        K *= inputTensorInfo.GetShape()[i];
    }

    for (unsigned int n = 0; n < inputTensorInfo.GetShape()[0]; n++)
    {
        for (unsigned int channelOutput = 0; channelOutput < N; channelOutput++)
        {
            float outval = 0.f;

            for (unsigned int channelInput = 0; channelInput < K; channelInput++)
            {
                float weight;
                if (transposeWeights)
                {
                    weight = weightData[channelOutput * K + channelInput];
                }
                else
                {
                    weight = weightData[channelInput * N + channelOutput];
                }

                outval += weight * inputData[n * K + channelInput];
            }

            if (biasData)
            {
                outval += biasData[channelOutput];
            }

            outputData[n * N + channelOutput] = outval;
        }
    }
}

} //namespace armnn
