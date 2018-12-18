//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "WorkloadUtils.hpp"

namespace armnn
{

armnn::ConstTensor PermuteTensor(const ConstCpuTensorHandle* tensor,
                                 const PermutationVector& permutationVector,
                                 void* permuteBuffer)
{
    BOOST_ASSERT_MSG(tensor, "Invalid input tensor");
    BOOST_ASSERT_MSG(permuteBuffer, "Invalid permute buffer");

    TensorInfo tensorInfo = tensor->GetTensorInfo();

    if (permutationVector.GetSize() > 0)
    {
        tensorInfo = armnnUtils::Permuted(tensorInfo, permutationVector);
        armnnUtils::Permute(tensorInfo.GetShape(), permutationVector,
                            tensor->GetConstTensor<void>(), permuteBuffer,
                            GetDataTypeSize(tensorInfo.GetDataType()));
    }
    else
    {
        ::memcpy(permuteBuffer, tensor->GetConstTensor<void>(), tensorInfo.GetNumBytes());
    }

    return ConstTensor(tensorInfo, permuteBuffer);
}

void ReshapeWeightsForAcl(TensorInfo& weightInfo, DataLayout dataLayout)
{
    // Reshape the weights in-place
    const TensorShape& weightShape = weightInfo.GetShape();
    switch (dataLayout)
    {
        case DataLayout::NHWC:
            // The data layout is NHWC, reshape from [ H, W, I, M ] to [ 1, H, W, I * M ]
            weightInfo.SetShape({ 1,
                                  weightShape[0],
                                  weightShape[1],
                                  weightShape[2] * weightShape[3] });
            break;
        case DataLayout::NCHW:
        default:
            // The data layout is NCHW, reshape from [ M, I, H, W ] to [ 1, I * M, H, W, ]
            weightInfo.SetShape({ 1,
                                  weightShape[0] * weightShape[1],
                                  weightShape[2],
                                  weightShape[3] });
            break;
    }
}

TensorInfo ConvertWeightTensorInfoFromArmnnToAcl(const TensorInfo& weightInfo, DataLayout dataLayout)
{
    // Convert the weight format from ArmNN's [ M, I, H, W ] (does NOT depend on the data layout) to either
    // [ 1, H, W, I * M ] (if NHWC) or [ 1, I * M, H, W ] (if NCHW), as required by the compute library

    // 1. Permute the weights if necessary
    // If the data layout is NCHW no permutation is necessary, as a reshape to [ 1, I * M, H, W ] can be better done
    // starting from the current shape of [ M, I, H, W ]
    TensorInfo weightPermutedInfo(weightInfo);
    if (dataLayout == DataLayout::NHWC)
    {
        // The data layout is NHWC, then permute the weights from [ M, I, H, W ] to [ H, W, I, M ]
        PermutationVector permutationVector{ 3, 2, 0, 1 };
        weightPermutedInfo = armnnUtils::Permuted(weightInfo, permutationVector);
    }

    // 2. Reshape the weights
    ReshapeWeightsForAcl(weightPermutedInfo, dataLayout);

    // 3. Return the permuted weight info
    return weightPermutedInfo;
}

armnn::ConstTensor ConvertWeightTensorFromArmnnToAcl(const ConstCpuTensorHandle* weightTensor,
                                                     DataLayout dataLayout,
                                                     void* permuteBuffer)
{
    BOOST_ASSERT_MSG(weightTensor, "Invalid input tensor");
    BOOST_ASSERT_MSG(permuteBuffer, "Invalid permute buffer");

    // Convert the weight format from ArmNN's [ M, I, H, W ] (does NOT depend on the data layout) to either
    // [ 1, H, W, I * M ] (if NHWC) or [ 1, I * M, H, W ] (if NCHW), as required by the compute library

    // 1. Permute the weights if necessary
    // If the data layout is NCHW no permutation is necessary, as a reshape to [ 1, I * M, H, W ] can be better done
    // starting from the current shape of [ M, I, H, W ]
    // If no permutation is necessary, leave the permutation vector empty
    PermutationVector permutationVector{};
    if (dataLayout == DataLayout::NHWC)
    {
        // The data layout is NHWC, then permute the weights from [ M, I, H, W ] to [ H, W, I, M ]
        permutationVector = { 3, 2, 0, 1 };
    }
    ConstTensor weightPermuted = PermuteTensor(weightTensor, permutationVector, permuteBuffer);

    // 2. Reshape the weights
    ReshapeWeightsForAcl(weightPermuted.GetInfo(), dataLayout);

    // 3. Return both the tensor and the allocated storage to ensure that the data stays alive
    return weightPermuted;
}

} // namespace armnn
