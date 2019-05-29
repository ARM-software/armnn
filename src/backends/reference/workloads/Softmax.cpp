//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Softmax.hpp"

#include <cmath>
#include <vector>

namespace armnn
{

/// Computes the softmax function on some inputs, into outputs, with a shape given by tensorInfo.
void Softmax(Decoder<float>& in, Encoder<float>& out, const TensorInfo& inputTensorInfo, float beta)
{
    unsigned int numChannels = inputTensorInfo.GetShape()[1];

    for (unsigned int n = 0; n < inputTensorInfo.GetShape()[0]; n++)
    {
        // Find maximum channel.
        in[n * numChannels];
        float max = in.Get();
        for (unsigned int c = 1; c < numChannels; c++)
        {
            in[n * numChannels + c];
            float val = in.Get();
            if (val > max)
            {
                max = val;
            }
        }

        // Exponentiate all values and sum.
        std::vector<float> exponentials(numChannels);
        float              sum = 0.0f;
        for (unsigned int c = 0; c < numChannels; c++)
        {
            in[n * numChannels + c];
            float val = in.Get();
            exponentials[c] = expf((val - max) * beta);
            sum += exponentials[c];
        }

        // Divide exponentials by sum to give outputs.
        for (unsigned int c = 0; c < numChannels; c++)
        {
            out[n * numChannels + c];
            out.Set(exponentials[c] / sum);
        }
    }
}

} //namespace armnn
