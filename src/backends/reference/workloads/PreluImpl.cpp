//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PreluImpl.hpp"
#include "RefWorkloadUtils.hpp"
#include "Broadcast.hpp"

namespace armnn
{

void PreluImpl(const PreluQueueDescriptor& data,
               Decoder<float>& inputData,
               Decoder<float>& alphaData,
               Encoder<float>& outputData)
{
    const TensorInfo& inputInfo  = GetTensorInfo(data.m_Inputs[0]);
    const TensorInfo& alphaInfo  = GetTensorInfo(data.m_Inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(data.m_Outputs[0]);

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
