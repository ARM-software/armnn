//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include "armnn/Exceptions.hpp"
#include <armnn/Descriptors.hpp>

namespace armnn
{
namespace armcomputetensorutils
{

arm_compute::DataType GetArmComputeDataType(armnn::DataType dataType)
{
    switch(dataType)
    {
        case armnn::DataType::Float16:
            return arm_compute::DataType::F16;
        case armnn::DataType::Float32:
            return arm_compute::DataType::F32;
        case armnn::DataType::QuantisedAsymm8:
            return arm_compute::DataType::QASYMM8;
        case armnn::DataType::QuantisedSymm16:
            return arm_compute::DataType::QSYMM16;
        case armnn::DataType::QuantisedSymm8:
            return arm_compute::DataType::QSYMM8;
        case armnn::DataType::Signed32:
            return arm_compute::DataType::S32;
        case armnn::DataType::Boolean:
            return arm_compute::DataType::U8;
        default:
            BOOST_ASSERT_MSG(false, "Unknown data type");
            return arm_compute::DataType::UNKNOWN;
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

// Utility function used to build a TensorInfo object, that can be used to initialise
// ARM Compute Tensor and CLTensor allocators.
arm_compute::TensorInfo BuildArmComputeTensorInfo(const armnn::TensorInfo& tensorInfo)
{
    const arm_compute::TensorShape aclTensorShape = BuildArmComputeTensorShape(tensorInfo.GetShape());
    const arm_compute::DataType aclDataType = GetArmComputeDataType(tensorInfo.GetDataType());
    const arm_compute::QuantizationInfo aclQuantizationInfo(tensorInfo.GetQuantizationScale(),
                                                            tensorInfo.GetQuantizationOffset());

    return arm_compute::TensorInfo(aclTensorShape, 1, aclDataType, aclQuantizationInfo);
}

arm_compute::TensorInfo BuildArmComputeTensorInfo(const armnn::TensorInfo& tensorInfo,
                                                  armnn::DataLayout dataLayout)
{
    const arm_compute::TensorShape aclTensorShape = BuildArmComputeTensorShape(tensorInfo.GetShape());
    const arm_compute::DataType aclDataType = GetArmComputeDataType(tensorInfo.GetDataType());
    const arm_compute::QuantizationInfo aclQuantizationInfo(tensorInfo.GetQuantizationScale(),
                                                            tensorInfo.GetQuantizationOffset());

    arm_compute::TensorInfo clTensorInfo(aclTensorShape, 1, aclDataType, aclQuantizationInfo);
    clTensorInfo.set_data_layout(ConvertDataLayout(dataLayout));

    return clTensorInfo;
}

arm_compute::DataLayout ConvertDataLayout(armnn::DataLayout dataLayout)
{
    switch(dataLayout)
    {
        case armnn::DataLayout::NHWC : return arm_compute::DataLayout::NHWC;

        case armnn::DataLayout::NCHW : return arm_compute::DataLayout::NCHW;

        default: throw InvalidArgumentException("Unknown armnn::DataLayout: [" +
                                                std::to_string(static_cast<int>(dataLayout)) + "]");
    }
}

arm_compute::PoolingLayerInfo BuildArmComputePoolingLayerInfo(const Pooling2dDescriptor& descriptor)
{
    using arm_compute::PoolingType;
    using arm_compute::DimensionRoundingType;
    using arm_compute::PadStrideInfo;
    using arm_compute::PoolingLayerInfo;
    using arm_compute::Size2D;

    // Resolve ARM Compute layer parameters.
    const PoolingType poolingType = ConvertPoolingAlgorithmToAclPoolingType(descriptor.m_PoolType);

    bool isGlobalPooling = (descriptor.m_StrideX==0 && descriptor.m_StrideY==0);
    //use specific constructor if global pooling
    if(isGlobalPooling)
    {
        return arm_compute::PoolingLayerInfo(poolingType);
    }

    const DimensionRoundingType rounding = ConvertOutputShapeRoundingToAclDimensionRoundingType(
                                                                                    descriptor.m_OutputShapeRounding);
    const PadStrideInfo padStrideInfo(descriptor.m_StrideX,
                                      descriptor.m_StrideY,
                                      descriptor.m_PadLeft,
                                      descriptor.m_PadRight,
                                      descriptor.m_PadTop,
                                      descriptor.m_PadBottom,
                                      rounding);

    const bool excludePadding = (descriptor.m_PaddingMethod == PaddingMethod::Exclude);

    const Size2D poolSize(descriptor.m_PoolWidth, descriptor.m_PoolHeight);

    return arm_compute::PoolingLayerInfo(poolingType, poolSize, padStrideInfo, excludePadding);
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

arm_compute::Size2D BuildArmComputeSize2D(const unsigned int width, const unsigned int height)
{
    return arm_compute::Size2D(width, height);
}

arm_compute::PixelValue GetPixelValue(arm_compute::ITensor& input, float pixelValue)
{
    switch (input.info()->data_type())
    {
        case arm_compute::DataType::QASYMM8:
            return arm_compute::PixelValue(static_cast<uint8_t>(pixelValue));
        case arm_compute::DataType::QSYMM16:
            return arm_compute::PixelValue(static_cast<int16_t>(pixelValue));
        case arm_compute::DataType::F16:
            return arm_compute::PixelValue(static_cast<Half>(pixelValue));
        case arm_compute::DataType::F32:
            return arm_compute::PixelValue(pixelValue);
        default:
            throw InvalidArgumentException("Unsupported DataType: [" +
                                           std::to_string(static_cast<int>(input.info()->data_type())) + "]");
    }
}

} // namespace armcomputetensorutils
} // namespace armnn
