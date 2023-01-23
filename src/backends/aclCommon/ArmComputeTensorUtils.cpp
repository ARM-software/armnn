//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include "armnn/Exceptions.hpp"
#include "ArmComputeUtils.hpp"
#include <armnn/Descriptors.hpp>

#include <fmt/format.h>

namespace armnn
{
namespace armcomputetensorutils
{

arm_compute::DataType GetArmComputeDataType(armnn::DataType dataType, bool multiScales)
{
    switch(dataType)
    {
        case armnn::DataType::BFloat16:
            return arm_compute::DataType::BFLOAT16;
        case armnn::DataType::Boolean:
            return arm_compute::DataType::U8;
        case armnn::DataType::Float16:
            return arm_compute::DataType::F16;
        case armnn::DataType::Float32:
            return arm_compute::DataType::F32;
        case armnn::DataType::QAsymmS8:
            return arm_compute::DataType::QASYMM8_SIGNED;
        case armnn::DataType::QAsymmU8:
            return arm_compute::DataType::QASYMM8;
        case armnn::DataType::QSymmS16:
            return arm_compute::DataType::QSYMM16;
        case armnn::DataType::Signed64:
            return arm_compute::DataType::S64;
        case armnn::DataType::QSymmS8:
        {
            return multiScales ? arm_compute::DataType::QSYMM8_PER_CHANNEL : arm_compute::DataType::QSYMM8;
        }
        case armnn::DataType::Signed32:
            return arm_compute::DataType::S32;
        default:
            ARMNN_ASSERT_MSG(false, "Unknown data type");
            return arm_compute::DataType::UNKNOWN;
    }
}

armnn::DataType GetArmNNDataType(arm_compute::DataType dataType)
{
    switch(dataType)
    {
        case arm_compute::DataType::BFLOAT16:
            return armnn::DataType::BFloat16;
        case arm_compute::DataType::U8:
            return armnn::DataType::Boolean;
        case arm_compute::DataType::F16:
            return armnn::DataType::Float16;
        case arm_compute::DataType::F32:
            return armnn::DataType::Float32;
        case arm_compute::DataType::QASYMM8_SIGNED:
            return armnn::DataType::QAsymmS8;
        case arm_compute::DataType::QASYMM8:
            return armnn::DataType::QAsymmU8;
        case arm_compute::DataType::QSYMM16:
            return armnn::DataType::QSymmS16;
        case arm_compute::DataType::S64:
            return armnn::DataType::Signed64;
        case arm_compute::DataType::QSYMM8_PER_CHANNEL:
            return armnn::DataType::QSymmS8;
        case arm_compute::DataType::QSYMM8:
            return armnn::DataType::QSymmS8;
        case arm_compute::DataType::S32:
            return armnn::DataType::Signed32;
        default:
            ARMNN_ASSERT_MSG(false, "Unknown data type");
            return armnn::DataType::Float32;
    }
}

arm_compute::Coordinates BuildArmComputeReductionCoordinates(size_t inputDimensions,
                                                             unsigned int originalInputRank,
                                                             const std::vector<unsigned int>& armnnAxes)
{
    arm_compute::Coordinates outAclCoords;

    if (armnnAxes.empty())
    {
        // If no reduction axes were provided, then the input must be reduced along all dimensions.
        // Since Compute Library does not accept an empty vector as the reduction dimensions, we then
        // manually create a vector including all the input dimensions (in reversed order) as:
        //
        // { inputDimensions - 1, inputDimensions - 2, ..., 1, 0 }
        //
        outAclCoords.set_num_dimensions(inputDimensions);
        std::generate(outAclCoords.begin(), outAclCoords.end(), [d = inputDimensions - 1] () mutable { return d--; });
    }
    else
    {
        // Create a vector of reduction dimensions (in reversed order) with the given reduction axes.
        //
        // Adjust the given reduction axes according to the original rank of the input tensor (before ACL applied any
        // dimension correction).
        // For example, if the input tensor originally had 4 dimensions, and one of the reduction axes was 2, then the
        // new value for that reduction axis should be 1.
        //
        // Example:
        // ArmNN input shape = { 1, 1, 3, 2 } -> ACL input shape = { 2, 3 }
        // ArmNN reduction axis = { 2 }       -> ACL reduction axis = { 1 }
        // ArmNN reduction axis = { 3 }       -> ACL reduction axis = { 0 }
        //
        // The transformation: ACL reduction axis index = original rank - ArmNN reduction axis index - 1
        //
        outAclCoords.set_num_dimensions(armnnAxes.size());
        std::transform(armnnAxes.begin(), armnnAxes.end(),
                       outAclCoords.begin(),
                       [originalInputRank](unsigned int i){ return originalInputRank - i - 1; });
    }

    return outAclCoords;
}

arm_compute::TensorShape BuildArmComputeTensorShape(const armnn::TensorShape& tensorShape)
{
    arm_compute::TensorShape shape;

    // armnn tensors are (batch, channels, height, width).
    // arm_compute tensors are (width, height, channels, batch).
    for (unsigned int i = 0; i < tensorShape.GetNumDimensions(); i++)
    {
        // Note that our dimensions are stored in the opposite order to ACL's.
        shape.set(tensorShape.GetNumDimensions() - i - 1, tensorShape[i], false);

        // TensorShape::set() flattens leading ones, so that batch size 1 cannot happen.
        // arm_compute tensors expect this.
    }

    // prevent arm_compute issue where tensor is flattened to nothing
    if (shape.num_dimensions() == 0)
    {
        shape.set_num_dimensions(1);
    }

    return shape;
}

std::vector<unsigned int> ReduceDimsForACL(const armnn::TensorShape tensorShape, unsigned int dimensions)
{
    std::vector<unsigned int> newShape;

    unsigned int dimsToSkip = 0;

    if (tensorShape.GetNumDimensions() > dimensions)
    {
        dimsToSkip = tensorShape.GetNumDimensions() - dimensions;
    }
    unsigned int dimsSkipped = 0;
    bool insertRemainder = false;

    for (unsigned int i = 0; i < tensorShape.GetNumDimensions(); ++i)
    {
        if (tensorShape[i] == 1 && dimsSkipped < dimsToSkip && !insertRemainder)
        {
            ++dimsSkipped;
            continue;
        }
        newShape.insert(newShape.begin(), tensorShape[i]);
        // Once we insert the first dimension we can't skip any more
        insertRemainder = true;
    }
    return newShape;
}

arm_compute::TensorShape BuildArmComputeTensorShape(const armnn::TensorShape& tensorShape, unsigned int dimensions)
{
    arm_compute::TensorShape shape;
    std::vector<unsigned int> strippedShape = ReduceDimsForACL(tensorShape, dimensions);

    for (unsigned int i = 0; i < strippedShape.size(); i++)
    {
        shape.set(i, strippedShape[i], false);
    }

    // prevent arm_compute issue where tensor is flattened to nothing
    if (shape.num_dimensions() == 0)
    {
        shape.set_num_dimensions(1);
    }
    return shape;
}

// Utility function used to build a TensorInfo object, that can be used to initialise
// ARM Compute Tensor and CLTensor allocators.
// Note: this utility ignores the value of armnn::TensorInfo.IsConstant(). ACL tensors
// default to constant but Arm NN ones default to non constant. In the cases where
// we expect ACL to treat a tensor as constant that value must be set after this
// utility has been called.
arm_compute::TensorInfo BuildArmComputeTensorInfo(const armnn::TensorInfo& tensorInfo)
{
    bool multiScales = tensorInfo.HasMultipleQuantizationScales();
    const arm_compute::TensorShape aclTensorShape = BuildArmComputeTensorShape(tensorInfo.GetShape());
    const arm_compute::DataType aclDataType       = GetArmComputeDataType(tensorInfo.GetDataType(), multiScales);

    const arm_compute::QuantizationInfo aclQuantizationInfo = multiScales ?
        arm_compute::QuantizationInfo(tensorInfo.GetQuantizationScales()) :
        arm_compute::QuantizationInfo(tensorInfo.GetQuantizationScale(), tensorInfo.GetQuantizationOffset());

    return arm_compute::TensorInfo(aclTensorShape, 1, aclDataType, aclQuantizationInfo);
}

arm_compute::TensorInfo BuildArmComputeTensorInfo(const armnn::TensorInfo& tensorInfo,
                                                  armnn::DataLayout dataLayout)
{
    arm_compute::TensorInfo aclTensorInfo = BuildArmComputeTensorInfo(tensorInfo);
    aclTensorInfo.set_data_layout(ConvertDataLayout(dataLayout));

    return aclTensorInfo;
}

arm_compute::TensorInfo BuildArmComputeTensorInfo(const armnn::TensorInfo& tensorInfo, unsigned int dimensions)
{
    bool multiScales = tensorInfo.HasMultipleQuantizationScales();
    const arm_compute::TensorShape aclTensorShape = BuildArmComputeTensorShape(tensorInfo.GetShape(), dimensions);
    const arm_compute::DataType aclDataType       = GetArmComputeDataType(tensorInfo.GetDataType(), multiScales);

    const arm_compute::QuantizationInfo aclQuantizationInfo = multiScales ?
              arm_compute::QuantizationInfo(tensorInfo.GetQuantizationScales()) :
              arm_compute::QuantizationInfo(tensorInfo.GetQuantizationScale(), tensorInfo.GetQuantizationOffset());

    return arm_compute::TensorInfo(aclTensorShape, 1, aclDataType, aclQuantizationInfo);
}
arm_compute::TensorInfo BuildArmComputeTensorInfo(const armnn::TensorInfo& tensorInfo,
                                                  armnn::DataLayout dataLayout, unsigned int dimensions)
{
    arm_compute::TensorInfo aclTensorInfo = BuildArmComputeTensorInfo(tensorInfo, dimensions);
    aclTensorInfo.set_data_layout(ConvertDataLayout(dataLayout));

    return aclTensorInfo;
}


arm_compute::DataLayout ConvertDataLayout(armnn::DataLayout dataLayout)
{
    switch(dataLayout)
    {
        case armnn::DataLayout::NHWC : return arm_compute::DataLayout::NHWC;

        case armnn::DataLayout::NCHW : return arm_compute::DataLayout::NCHW;

        case armnn::DataLayout::NDHWC : return arm_compute::DataLayout::NDHWC;

        case armnn::DataLayout::NCDHW : return arm_compute::DataLayout::NCDHW;

        default: throw InvalidArgumentException("Unknown armnn::DataLayout: [" +
                                                std::to_string(static_cast<int>(dataLayout)) + "]");
    }
}

arm_compute::PoolingLayerInfo BuildArmComputePoolingLayerInfo(const Pooling2dDescriptor& descriptor,
                                                              bool fpMixedPrecision)
{
    // Resolve ARM Compute layer parameters.
    const arm_compute::PoolingType poolingType = ConvertPoolingAlgorithmToAclPoolingType(descriptor.m_PoolType);

    const arm_compute::DataLayout dataLayout = ConvertDataLayout(descriptor.m_DataLayout);

    bool isGlobalPooling = (descriptor.m_StrideX==0 && descriptor.m_StrideY==0);
    //use specific constructor if global pooling
    if(isGlobalPooling)
    {
        return arm_compute::PoolingLayerInfo(poolingType, dataLayout);
    }

    const arm_compute::DimensionRoundingType rounding = ConvertOutputShapeRoundingToAclDimensionRoundingType(
                                                                                    descriptor.m_OutputShapeRounding);
    const arm_compute::PadStrideInfo padStrideInfo(descriptor.m_StrideX,
                                      descriptor.m_StrideY,
                                      descriptor.m_PadLeft,
                                      descriptor.m_PadRight,
                                      descriptor.m_PadTop,
                                      descriptor.m_PadBottom,
                                      rounding);

    const bool excludePadding = (descriptor.m_PaddingMethod == PaddingMethod::Exclude);

    const arm_compute::Size2D poolSize(descriptor.m_PoolWidth, descriptor.m_PoolHeight);

    return arm_compute::PoolingLayerInfo(poolingType, poolSize, dataLayout, padStrideInfo, excludePadding,
                                         fpMixedPrecision);
}

arm_compute::Pooling3dLayerInfo BuildArmComputePooling3dLayerInfo(const Pooling3dDescriptor& descriptor,
                                                                  bool fpMixedPrecision)
{
    const arm_compute::PoolingType poolingType = ConvertPoolingAlgorithmToAclPoolingType(descriptor.m_PoolType);

    bool isGlobalPooling = (descriptor.m_StrideX==0 && descriptor.m_StrideY==0 && descriptor.m_StrideZ==0);
    //use specific constructor if global pooling
    if(isGlobalPooling)
    {
        return arm_compute::Pooling3dLayerInfo(poolingType);
    }

    const arm_compute::Size3D poolSize(descriptor.m_PoolWidth, descriptor.m_PoolHeight, descriptor.m_PoolDepth);

    const arm_compute::Size3D stride(descriptor.m_StrideX,
                        descriptor.m_StrideY,
                        descriptor.m_StrideZ);

    const arm_compute::Padding3D padding(descriptor.m_PadLeft,
                            descriptor.m_PadRight,
                            descriptor.m_PadTop,
                            descriptor.m_PadBottom,
                            descriptor.m_PadFront,
                            descriptor.m_PadBack);

    const bool excludePadding = (descriptor.m_PaddingMethod == PaddingMethod::Exclude);

    const arm_compute::DimensionRoundingType rounding = ConvertOutputShapeRoundingToAclDimensionRoundingType(
            descriptor.m_OutputShapeRounding);

    return arm_compute::Pooling3dLayerInfo(poolingType,
                                           poolSize,
                                           stride,
                                           padding,
                                           excludePadding,
                                           fpMixedPrecision,
                                           rounding);
}

arm_compute::NormalizationLayerInfo BuildArmComputeNormalizationLayerInfo(const NormalizationDescriptor& descriptor)
{
    const arm_compute::NormType normType =
        ConvertNormalizationAlgorithmChannelToAclNormType(descriptor.m_NormChannelType);
    return arm_compute::NormalizationLayerInfo(normType,
                                               descriptor.m_NormSize,
                                               descriptor.m_Alpha,
                                               descriptor.m_Beta,
                                               descriptor.m_K,
                                               false);
}

arm_compute::PermutationVector BuildArmComputePermutationVector(const armnn::PermutationVector& perm)
{
    arm_compute::PermutationVector aclPerm;

    unsigned int start = 0;
    while ((start < perm.GetSize()) && (start == perm[start]))
    {
        ++start;
    }

    for (unsigned int i = start; i < perm.GetSize(); ++i)
    {
        aclPerm.set(i - start, perm[i] - start);
    }
    return aclPerm;
}

arm_compute::PermutationVector BuildArmComputeTransposeVector(const armnn::PermutationVector& perm)
{
    // As ArmNN indexes are left to right and ACL indexes are right to left,
    // the permutation vector has to be reversed and then translated into ACL axis.
    // i.e. {1, 0, 2, 3} --> {3, 2, 0, 1} --> {0, 1, 3, 2}

    // Below an example of how the ArmNN and ACL index format work:
    // ArmNN Format:
    // Input Shape        {1, 10, 20, 30}
    // Permutation Vector {1,  0,  2,  3}
    // Output Shape       {10, 1, 20, 30}
    // dim "1" of input goes into index 0 of the output ([ 10, X, X, X])
    // dim "0" of input goes into index 1 of the output ([ 10, 1, X, X ])
    // dim "2" of input goes into index 2 of the output ([ 10, 1, 20, X ])
    // dim "3" of input goes into index 3 of the output ([ 10, 1, 20, 30 ])
    // ACL Format:
    // Input Shape        {30, 20, 10, 1}
    // Permutation Vector {0,  1,  3,  2}
    // Output Shape       {30, 20, 1, 10}
    // dim "0" of input goes into index 0 of the output ([ 30,  X, X, X])
    // dim "1" of input goes into index 1 of the output ([ 30, 20, X, X ])
    // dim "3" of input goes into index 2 of the output ([ 30, 20, 1, X ])
    // dim "2" of input goes into index 3 of the output ([ 30, 20, 1, 10 ])

    arm_compute::PermutationVector aclPerm;
    auto rank = perm.GetSize();

    // Reverse the order. i.e. {1, 0, 2, 3} --> {3, 2, 0, 1}
    std::vector<unsigned int> reversedPerm;
    reversedPerm.reserve(rank);
    for (unsigned int i = rank; i > 0; --i)
    {
        reversedPerm.push_back(perm[i-1]);
    }

    // Translate from Arm NN axis to ACL axis. i.e. {3, 2, 0, 1} --> {0, 1, 3, 2}
    for (unsigned int i = 0; i < rank; ++i)
    {
        auto aclAxis = rank - 1 - reversedPerm[i];
        aclPerm.set(i, aclAxis);
    }
    return aclPerm;
}

arm_compute::Size2D BuildArmComputeSize2D(const unsigned int width, const unsigned int height)
{
    return arm_compute::Size2D(width, height);
}

arm_compute::PixelValue GetPixelValue(const arm_compute::ITensorInfo* tensorInfo, float value)
{
    switch (tensorInfo->data_type())
    {
        case arm_compute::DataType::F16:
        {
            arm_compute::PixelValue pixelValue = arm_compute::PixelValue(static_cast<Half>(value));
            if (isinf(pixelValue.get<Half>())) {
                throw InvalidArgumentException("Under/Overflow converting float value [" + std::to_string(value) +
                    "] to fp16: [" + std::to_string(pixelValue.get<Half>()) + "]");
            }
            return pixelValue;
        }
        case arm_compute::DataType::F32:
            return arm_compute::PixelValue(value);
        case arm_compute::DataType::QASYMM8:
            return arm_compute::PixelValue(static_cast<uint8_t>(value));
        case arm_compute::DataType::QSYMM16:
            return arm_compute::PixelValue(static_cast<int16_t>(value));
        case arm_compute::DataType::QSYMM8:
        case arm_compute::DataType::QASYMM8_SIGNED:
        case arm_compute::DataType::QSYMM8_PER_CHANNEL:
            return arm_compute::PixelValue(static_cast<int8_t>(value));
        case arm_compute::DataType::S32:
            return arm_compute::PixelValue(static_cast<int32_t>(value));
        default:
            throw InvalidArgumentException("Unsupported DataType: [" +
                                           std::to_string(static_cast<int>(tensorInfo->data_type())) + "]");
    }
}

unsigned int ComputeDepthwiseConv2dDepthMultiplier(armnn::DataLayout layout,
                                                   const arm_compute::TensorShape& weightsShape,
                                                   const arm_compute::TensorShape& inputShape)
{
    unsigned int depthMultiplier;
    if (layout == armnn::DataLayout::NHWC)
    {
        depthMultiplier = static_cast<uint32_t>(weightsShape[0]) / static_cast<uint32_t>(inputShape[0]);
    }
    else if (layout == armnn::DataLayout::NCHW)
    {
        depthMultiplier = static_cast<uint32_t>(weightsShape[2]) / static_cast<uint32_t>(inputShape[2]);
    }
    else
    {
        throw InvalidArgumentException(fmt::format("Unknown data layout for tensor conversion: {}",
                                                   GetDataLayoutName(layout)));
    }
    return depthMultiplier;
}

} // namespace armcomputetensorutils
} // namespace armnn
