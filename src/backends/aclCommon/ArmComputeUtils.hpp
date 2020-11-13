//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Descriptors.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/utility/Assert.hpp>
#include <backendsCommon/WorkloadData.hpp>

#include <arm_compute/core/Types.h>

namespace armnn
{

inline arm_compute::NormalizationLayerInfo
CreateAclNormalizationLayerInfoForL2Normalization(const armnn::TensorInfo& tensorInfo,
                                                  armnn::DataLayout dataLayout)
{
    unsigned int depthDimension = dataLayout == armnn::DataLayout::NCHW ? 1 : 3;
    const unsigned int depth = tensorInfo.GetShape()[depthDimension];

    // At the time of writing, {CL|Neon}L2Normalization performs the reduction only along dimension 0. This version of
    // L2 Normalization always performs the reduction along the depth axis, though. Thus, we repurpose
    // {CL|Neon}NormalizationLayers to act as depthwise L2 normalizations by carefully chosing the normalization
    // parameters.
    //
    // Please refer to both the reference implementation of the normalization layer and the implementation of
    // {CL|Neon}NormalizationLayer when checking the derivations for the parameter values below.

    // Make sure normalization covers the entire depth range. ACL requires the normalization size to be odd.
    // CL: This does not result in extra kernel threads not doing any work: See usage of the RADIUS parameter in
    // ACL's normalization_layer_cross_map() CL function.
    const uint32_t normSize = depth * 2u + 1u;

    // See ACL's NormalizationLayerInfo::scale_coeff() definition.
    // For the reference implementation, to make alpha_ become 1, we'd have to use alpha = normSize instead.
    const float alpha = 1.0f;

    // Don't offset the reduction.
    const float kappa = 0.0f;

    // pow(reduction, -0.5) = 1 / sqrt(reduction)
    const float beta = 0.5f;

    return arm_compute::NormalizationLayerInfo(arm_compute::NormType::CROSS_MAP, normSize, alpha, beta, kappa, false);
}

inline arm_compute::ActivationLayerInfo::ActivationFunction
ConvertActivationFunctionToAclActivationFunction(ActivationFunction armnnFunction)
{
    using AclActivationFunction = arm_compute::ActivationLayerInfo::ActivationFunction;

    switch (armnnFunction)
    {
        case ActivationFunction::Linear:        return AclActivationFunction::LINEAR;
        // Arm compute's 'logistic' function is non-parameterized, so it is exactly a sigmoid function.
        case ActivationFunction::Sigmoid:       return AclActivationFunction::LOGISTIC;
        case ActivationFunction::ReLu:          return AclActivationFunction::RELU;
        case ActivationFunction::BoundedReLu:   return AclActivationFunction::LU_BOUNDED_RELU;
        case ActivationFunction::SoftReLu:      return AclActivationFunction::SOFT_RELU;
        case ActivationFunction::LeakyReLu:     return AclActivationFunction::LEAKY_RELU;
        case ActivationFunction::Abs:           return AclActivationFunction::ABS;
        case ActivationFunction::Sqrt:          return AclActivationFunction::SQRT;
        case ActivationFunction::Square:        return AclActivationFunction::SQUARE;
        case ActivationFunction::TanH:          return AclActivationFunction::TANH;
        case ActivationFunction::Elu:           return AclActivationFunction::ELU;
        case ActivationFunction::HardSwish:     return AclActivationFunction::HARD_SWISH;
        default:                                throw InvalidArgumentException("Unsupported activation function");
    }
}

inline arm_compute::ActivationLayerInfo
ConvertActivationDescriptorToAclActivationLayerInfo(const ActivationDescriptor& actDesc)
{
    return arm_compute::ActivationLayerInfo(ConvertActivationFunctionToAclActivationFunction(actDesc.m_Function),
        actDesc.m_A, actDesc.m_B);
}

inline arm_compute::ActivationLayerInfo
ConvertActivationDescriptorToAclActivationLayerInfo(const ActivationDescriptor* activationDescPtr)
{
    if (activationDescPtr != nullptr)
    {
        return ConvertActivationDescriptorToAclActivationLayerInfo(static_cast<ActivationDescriptor>(
                                                                           *activationDescPtr));
    }
    return arm_compute::ActivationLayerInfo();
}

inline arm_compute::ActivationLayerInfo
ConvertAdditionalInfoToAclActivationLayerInfo(const QueueDescriptor& queueDescriptor)
{
    const ActivationDescriptor* activationDescPtr = queueDescriptor.GetAdditionalInformation<ActivationDescriptor>();

    if (activationDescPtr != nullptr)
    {
        return ConvertActivationDescriptorToAclActivationLayerInfo(static_cast<ActivationDescriptor>(
                *activationDescPtr));
    }
    return arm_compute::ActivationLayerInfo();
}

inline arm_compute::ComparisonOperation ConvertComparisonOperationToAcl(const ComparisonDescriptor& descriptor)
{
    switch (descriptor.m_Operation)
    {
        case ComparisonOperation::Greater:         return arm_compute::ComparisonOperation::Greater;
        case ComparisonOperation::GreaterOrEqual:  return arm_compute::ComparisonOperation::GreaterEqual;
        case ComparisonOperation::Less:            return arm_compute::ComparisonOperation::Less;
        case ComparisonOperation::LessOrEqual:     return arm_compute::ComparisonOperation::LessEqual;
        case ComparisonOperation::Equal:           return arm_compute::ComparisonOperation::Equal;
        case ComparisonOperation::NotEqual:        return arm_compute::ComparisonOperation::NotEqual;
        default:                                   throw InvalidArgumentException("Unsupported comparison function");
    }
}

inline arm_compute::PoolingType ConvertPoolingAlgorithmToAclPoolingType(PoolingAlgorithm poolingAlgorithm)
{
    using arm_compute::PoolingType;

    switch (poolingAlgorithm)
    {
        case PoolingAlgorithm::Max:             return PoolingType::MAX;
        case PoolingAlgorithm::Average:         return PoolingType::AVG;
        case PoolingAlgorithm::L2:              return PoolingType::L2;
        default:                                throw InvalidArgumentException("Unsupported pooling algorithm");
    }
}

inline arm_compute::DimensionRoundingType ConvertOutputShapeRoundingToAclDimensionRoundingType(OutputShapeRounding
                                                                                                              rounding)
{
    using arm_compute::DimensionRoundingType;

    switch (rounding)
    {
        case OutputShapeRounding::Ceiling:  return DimensionRoundingType::CEIL;
        case OutputShapeRounding::Floor:    return DimensionRoundingType::FLOOR;
        default:                            throw InvalidArgumentException("Unsupported Output Shape Rounding type");
    }
}

inline arm_compute::NormType
ConvertNormalizationAlgorithmChannelToAclNormType(NormalizationAlgorithmChannel channelType)
{
    using arm_compute::NormType;
    switch (channelType)
    {
        case NormalizationAlgorithmChannel::Across: return NormType::CROSS_MAP;
        case NormalizationAlgorithmChannel::Within: return NormType::IN_MAP_2D;
        default:    throw InvalidArgumentException("Unsupported normalization algorithm channel type");
    }
}

inline arm_compute::FullyConnectedLayerInfo
ConvertFullyConnectedDescriptorToAclFullyConnectedLayerInfo(const FullyConnectedDescriptor& fullyConnectedDesc,
                                                            const ActivationDescriptor* activationDesc)
{
    arm_compute::FullyConnectedLayerInfo fc_info;
    fc_info.transpose_weights = fullyConnectedDesc.m_TransposeWeightMatrix;
    fc_info.activation_info = ConvertActivationDescriptorToAclActivationLayerInfo(activationDesc);
    return fc_info;
}

inline arm_compute::FullyConnectedLayerInfo
ConvertFullyConnectedDescriptorToAclFullyConnectedLayerInfo(const FullyConnectedDescriptor& fullyConnectedDesc,
        arm_compute::ActivationLayerInfo activationLayerInfo)
{
    arm_compute::FullyConnectedLayerInfo fc_info;
    fc_info.transpose_weights = fullyConnectedDesc.m_TransposeWeightMatrix;
    fc_info.activation_info = activationLayerInfo;
    return fc_info;
}

inline arm_compute::InterpolationPolicy ConvertResizeMethodToAclInterpolationPolicy(ResizeMethod resizeMethod)
{
    switch (resizeMethod)
    {
        case ResizeMethod::Bilinear:
            return arm_compute::InterpolationPolicy::BILINEAR;
        case ResizeMethod::NearestNeighbor:
            return arm_compute::InterpolationPolicy::NEAREST_NEIGHBOR;
        default:
            throw InvalidArgumentException("Unsupported resize method");
    }
}

template<typename T>
inline T ComputeSoftmaxAclAxis(const SoftmaxDescriptor& softmaxDesc, const armnn::TensorInfo& tensor)
{
    // Detect the Android default value of -1 and return the ACL default value of 0.
    if (softmaxDesc.m_Axis == -1)
    {
        return 0;
    }

    unsigned int dim = tensor.GetNumDimensions();

    ARMNN_ASSERT(dim != 0);

    // Currently ArmNN support axis 1.
    auto aclAxis = (static_cast<T>(dim) - 1);
    aclAxis = aclAxis > 0 ? aclAxis -1 : aclAxis;

    return aclAxis;
}

inline std::set<unsigned int> ComputeSplitAxis(const armnn::SplitterDescriptor& desc, const TensorShape& input)
{
    unsigned int numSplit = desc.GetNumViews();
    unsigned int numDimensions = desc.GetNumDimensions();
    std::set<unsigned int> splitAxis;

    for (unsigned int i = 0; i < numSplit; ++i)
    {
        for (unsigned int dimIdx = 0; dimIdx < numDimensions; ++dimIdx)
        {
            if (desc.GetViewSizes(i)[dimIdx] != input[dimIdx])
            {
                splitAxis.insert(dimIdx);
            }
        }
    }
    return splitAxis;
}

/// Function to convert ArmNN axis (left to right) to ACL axis (right to left) ranging from [-rank, rank)
inline int ComputeAclAxis(const int& armnnAxis, const armnn::TensorInfo& tensor)
{
    int rank = static_cast<int>(tensor.GetNumDimensions());

    ARMNN_ASSERT(rank != 0);
    ARMNN_ASSERT((-1 * rank) <= armnnAxis);
    ARMNN_ASSERT(armnnAxis < rank);

    int sign = (armnnAxis < 0) ? -1 : 1;
    int aclAxis = sign * rank - 1  - armnnAxis;

    return aclAxis;
}

/// Function to convert axis to its positive equivalent value.
/// [-rank, rank) --> [0, rank)
inline unsigned int ComputePositiveAxis(const int& axis, const armnn::TensorInfo& tensor)
{
    int rank = static_cast<int>(tensor.GetNumDimensions());

    ARMNN_ASSERT(rank != 0);
    ARMNN_ASSERT((-1 * rank) <= axis);
    ARMNN_ASSERT(axis < rank);

    int positiveAxis = (axis < 0) ? rank + axis : axis;
    return static_cast<unsigned int>(positiveAxis);
}

} // namespace armnn
