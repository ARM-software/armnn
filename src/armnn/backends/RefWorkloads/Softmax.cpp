//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "Softmax.hpp"

#include <cmath>
#include <vector>

namespace armnn
{

/// Computes the softmax function on some inputs, into outputs, with a shape given by tensorInfo
void Softmax(const float* in, float* out, const TensorInfo& tensorInfo, float beta)
{
    unsigned int numChannels = tensorInfo.GetShape()[1];
    for (unsigned int n = 0; n < tensorInfo.GetShape()[0]; n++)
    {
        // find maximum channel
        float max = in[n * numChannels];
        for (unsigned int c = 1; c < numChannels; c++)
        {
            float val = in[n * numChannels + c];
            if (val > max)
            {
                max = val;
            }
        }

        // exponentiate all values and sum
        std::vector<float> exponentials(numChannels);
        float              sum = 0.0f;
        for (unsigned int c = 0; c < numChannels; c++)
        {
            float val       = in[n * numChannels + c];
            exponentials[c] = expf((val - max) * beta);
            sum += exponentials[c];
        }

        // divide exponentials by sum to give outputs
        for (unsigned int c = 0; c < numChannels; c++)
        {
            out[n * numChannels + c] = exponentials[c] / sum;
        }
    }
}

} //namespace armnn
