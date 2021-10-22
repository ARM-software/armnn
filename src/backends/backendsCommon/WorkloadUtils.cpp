//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/WorkloadUtils.hpp>

#include <armnn/Utils.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnnUtils/DataLayoutIndexed.hpp>

#include <fmt/format.h>

namespace armnn
{

armnn::ConstTensor PermuteTensor(const ConstTensorHandle* tensor,
                                 const PermutationVector& permutationVector, void* permuteBuffer)
{
    ARMNN_ASSERT_MSG(tensor, "Invalid input tensor");
    ARMNN_ASSERT_MSG(permuteBuffer, "Invalid permute buffer");

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
    tensorInfo.SetConstant(true);
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
            weightInfo.SetShape({ 1,
                                  weightShape[0] * weightShape[1],
                                  weightShape[2],
                                  weightShape[3] });
            break;
        case DataLayout::NCHW:
        default:
            // The data layout is NCHW, reshape from [ M, I, H, W ] to [ 1, I * M, H, W, ]
            weightInfo.SetShape({ 1, weightShape[0] * weightShape[1], weightShape[2], weightShape[3] });
            break;
    }
}

template <typename DataType>
ConstTensor ReorderWeightChannelsForAcl(const ConstTensor& weightHandle, DataLayout dataLayout, void* permuteBuffer)
{
    DataType* weight = static_cast<DataType*>(permuteBuffer);
    const TensorShape& weightShape = weightHandle.GetShape();
    unsigned int multiplier;
    unsigned int height;
    unsigned int width;
    unsigned int inputChannels;
    switch (dataLayout)
    {
        case DataLayout::NHWC:    //It actually is [ H, W, I, M ]
            height        = weightShape[0];
            width         = weightShape[1];
            inputChannels = weightShape[2];
            multiplier    = weightShape[3];
            break;
        case DataLayout::NCHW:    //It actually is [ M, I, H, W ]
        default:
            height        = weightShape[2];
            width         = weightShape[3];
            inputChannels = weightShape[1];
            multiplier    = weightShape[0];
            break;
    }

    std::vector<DataType> weightAclOrder(height*width*inputChannels*multiplier);
    unsigned int destinationWeightsChannel;
    unsigned int totalChannels = inputChannels * multiplier;
    unsigned int channelSize   = height * width;
    unsigned int inputChannel  = 0;

    for (unsigned int originWeightsChannel = 0; originWeightsChannel < totalChannels; originWeightsChannel++)
    {
        inputChannel = originWeightsChannel % inputChannels;
        destinationWeightsChannel = (originWeightsChannel - inputChannel) / inputChannels + multiplier * inputChannel;

        for (unsigned int i = 0; i < channelSize; i++)
        {
            weightAclOrder[i + destinationWeightsChannel * channelSize] =
                    weight[i + originWeightsChannel * channelSize];
        }
    }

    ::memcpy(permuteBuffer, weightAclOrder.data(), weightHandle.GetInfo().GetNumBytes());
    return ConstTensor(weightHandle.GetInfo(), permuteBuffer);
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


std::tuple<ConstTensor, unsigned int> Convert1HWOTensorToAcl(const ConstTensorHandle* weightTensor,
                                                             const TensorInfo& inputInfo,
                                                             const DataLayout dataLayout,
                                                             void* permuteBuffer)
{
    TensorInfo weightsInfo = weightTensor->GetTensorInfo();
    unsigned int depthMultiplier = 1;
    PermutationVector permutationVector{};
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        // No permutation required. Data layouts are the same.

        depthMultiplier = weightsInfo.GetShape()[3] / inputInfo.GetShape()[3];
    }
    else if (dataLayout == armnn::DataLayout::NCHW)
    {
        // [ 1, H, W, I*M] --> [ 1, I * M, H, W ]
        depthMultiplier = weightsInfo.GetShape()[3] / inputInfo.GetShape()[1];
        permutationVector = { 0, 2, 3, 1 };
    }
    else
    {
        throw InvalidArgumentException(fmt::format("Unknown data layout for tensor conversion: {}",
                                                   GetDataLayoutName(dataLayout)));
    }

    ConstTensor weightsPermuted = PermuteTensor(weightTensor, permutationVector, permuteBuffer);

    return std::make_tuple(weightsPermuted, depthMultiplier);
}

std::tuple<TensorInfo, unsigned int> Convert1HWOTensorInfoToAcl(const TensorInfo& weightInfo,
                                                                const TensorInfo& inputInfo,
                                                                const DataLayout dataLayout)
{
    unsigned int aclDepthMultiplier = 1;
    TensorInfo weightsPermuted;
    if (dataLayout == armnn::DataLayout::NHWC)
    {
        // No permutation required. Data layouts are the same.
        aclDepthMultiplier = weightInfo.GetShape()[3] / inputInfo.GetShape()[3];
        weightsPermuted = weightInfo;
    }
    else if (dataLayout == armnn::DataLayout::NCHW)
    {
        // [ 1, H, W, I*M] --> [ 1, I * M, H, W ]
        aclDepthMultiplier = weightInfo.GetShape()[3] / inputInfo.GetShape()[1];
        PermutationVector permutationVector{ 0, 2, 3, 1 };
        weightsPermuted = armnnUtils::Permuted(weightInfo, permutationVector);
    }
    else
    {
        throw InvalidArgumentException(fmt::format("Unknown data layout for tensor info conversion: {}",
                                                   GetDataLayoutName(dataLayout)));
    }

    return std::make_tuple(weightsPermuted, aclDepthMultiplier);
}


std::tuple<ConstTensor, unsigned int> Convert1HWOtoMIHW(const ConstTensorHandle* weightTensor,
                                                        const TensorInfo& inputInfo,
                                                        const DataLayout& dataLayout,
                                                        void* permuteBuffer)
{
    TensorInfo weightsInfo = weightTensor->GetTensorInfo();

    if (weightsInfo.HasPerAxisQuantization())
    {
        throw InvalidArgumentException("Can't convert tensor from [1,H,W,Cout] to [M,Cin,H,W] when per channel "
                                       "quantization is applied.");
    }

    // Reshape weights  [ 1, H, W, I*M ] --> [ H, W, I, M ]
    auto weightsShape = weightsInfo.GetShape();
    auto channelIndex = armnnUtils::DataLayoutIndexed(dataLayout).GetChannelsIndex();
    unsigned int depthMultiplier = weightsShape[3] / inputInfo.GetShape()[channelIndex];
    weightsInfo.SetShape({ weightsShape[1],
                           weightsShape[2],
                           inputInfo.GetShape()[channelIndex],
                           depthMultiplier});

    // Permute [ H, W, I, M ] --> [ M, I, H, W ]
    PermutationVector permutationVector = { 2, 3, 1, 0 };
    ConstTensor weightsPermuted = PermuteTensor(weightTensor, permutationVector, permuteBuffer);

    return std::make_tuple(weightsPermuted, depthMultiplier);
}

armnn::ConstTensor ConvertWeightTensorFromArmnnToAcl(const ConstTensorHandle* weightTensor,
                                                     DataLayout dataLayout,
                                                     void* permuteBuffer)
{
    ARMNN_ASSERT_MSG(weightTensor, "Invalid input tensor");
    ARMNN_ASSERT_MSG(permuteBuffer, "Invalid permute buffer");

    auto multiplier    = weightTensor->GetTensorInfo().GetShape()[0];
    auto inputChannels = weightTensor->GetTensorInfo().GetShape()[1];

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

    // Shuffle the weights data to obtain the channel order needed used by Acl
    if (multiplier > 1 && inputChannels > 1 && dataLayout == DataLayout::NCHW)
    {
        switch (weightPermuted.GetDataType())
        {
            case DataType::Float32:
                weightPermuted = ReorderWeightChannelsForAcl<float>(weightPermuted, dataLayout, permuteBuffer);
                break;
            case DataType::Float16:
                weightPermuted =
                    ReorderWeightChannelsForAcl<half_float::half>(weightPermuted, dataLayout, permuteBuffer);
                break;
            case DataType::QAsymmS8:
            case DataType::QAsymmU8:
                weightPermuted = ReorderWeightChannelsForAcl<uint8_t>(weightPermuted, dataLayout, permuteBuffer);
                break;
            case DataType::QSymmS8:
                weightPermuted = ReorderWeightChannelsForAcl<int8_t>(weightPermuted, dataLayout, permuteBuffer);
                break;
            default:
                break;
        }
    }

    // 2. Reshape the weights
    ReshapeWeightsForAcl(weightPermuted.GetInfo(), dataLayout);

    // 3. Return both the tensor and the allocated storage to ensure that the data stays alive
    return weightPermuted;
}

int32_t ConvertMaskToACLFormat(int32_t mask, int32_t numDim)
{
    int32_t reversedMask = 0;
    for (unsigned int i = 0; i < armnn::numeric_cast<unsigned int>(numDim); ++i)
    {
        // Check if bit set in mask for each dimension
        int32_t bit = (mask & 1 << i) != 0;
        // Increment the new mask with the bits reversed
        reversedMask += (bit << std::max(numDim-(armnn::numeric_cast<int>(i)+1), 0));
    }

    return reversedMask;
}

} // namespace armnn
