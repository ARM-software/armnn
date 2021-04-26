//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PreluImpl.hpp"
#include "RefWorkloadUtils.hpp"
#include "Broadcast.hpp"

namespace armnn
{

void PreluImpl(const TensorInfo& inputInfo,
               const TensorInfo& alphaInfo,
               const TensorInfo& outputInfo,
               Decoder<float>& inputData,
               Decoder<float>& alphaData,
               Encoder<float>& outputData)
{
    const TensorShape& inputShape  = inputInfo.GetShape();
    const TensorShape& alphaShape  = alphaInfo.GetShape();
    const TensorShape& outputShape = outputInfo.GetShape();

    // PReLU activation: f(x) = alpha * x for x < 0, f(x) = x for x >= 0
    auto prelu = [](float x, float alpha)
    {
        return x < 0 ? alpha * x : x;
    };

    BroadcastLoop(inputShape, alphaShape, outputShape).Unroll(prelu, 0, inputData, alphaData, outputData);
}

} // namespace armnn
