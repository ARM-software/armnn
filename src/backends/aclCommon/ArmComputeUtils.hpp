//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Descriptors.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/backends/WorkloadData.hpp>

#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/FunctionDescriptors.h>

#if defined(ARMCOMPUTENEON_ENABLED)
#include "neon/workloads/NeonReduceWorkload.hpp"
#endif

#if defined(ARMCOMPUTECL_ENABLED)
#include "cl/workloads/ClReduceWorkload.hpp"
#endif

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

/// Utility function used to setup an arm_compute::Conv3dInfo object from convolution3d descriptor.
inline arm_compute::Conv3dInfo ComputeConv3DInfo(const armnn::Convolution3dDescriptor descriptor,
                                                 bool isFastMathEnabled,
                                                 const ActivationDescriptor* activationDescriptor)
{
    const arm_compute::Size3D    stride{descriptor.m_StrideX, descriptor.m_StrideY, descriptor.m_StrideZ};
    const arm_compute::Padding3D padding{descriptor.m_PadLeft, descriptor.m_PadRight,
                                         descriptor.m_PadTop, descriptor.m_PadBottom,
                                         descriptor.m_PadFront, descriptor.m_PadBack};
    const arm_compute::Size3D    dilation{descriptor.m_DilationX, descriptor.m_DilationY, descriptor.m_DilationZ};

    const arm_compute::ActivationLayerInfo activationInfo =
            ConvertActivationDescriptorToAclActivationLayerInfo(activationDescriptor);
    const auto roundType = arm_compute::DimensionRoundingType::FLOOR;

    return arm_compute::Conv3dInfo{stride, padding, activationInfo, dilation, roundType, isFastMathEnabled};
}

inline arm_compute::Conv3dInfo ComputeConv3DInfo(const armnn::Convolution3dQueueDescriptor queueDescriptor,
                                                 bool isFastMathEnabled)
{
    auto descriptor = queueDescriptor.m_Parameters;
    const arm_compute::Size3D    stride{descriptor.m_StrideX, descriptor.m_StrideY, descriptor.m_StrideZ};
    const arm_compute::Padding3D padding{descriptor.m_PadLeft, descriptor.m_PadRight,
                                         descriptor.m_PadTop, descriptor.m_PadBottom,
                                         descriptor.m_PadFront, descriptor.m_PadBack};
    const arm_compute::Size3D    dilation{descriptor.m_DilationX, descriptor.m_DilationY, descriptor.m_DilationZ};

    const arm_compute::ActivationLayerInfo activationInfo =
            ConvertAdditionalInfoToAclActivationLayerInfo(queueDescriptor);
    const auto roundType = arm_compute::DimensionRoundingType::FLOOR;

    return arm_compute::Conv3dInfo{stride, padding, activationInfo, dilation, roundType, isFastMathEnabled};
}

inline arm_compute::PaddingMode ConvertPaddingModeToAcl(const PaddingMode& paddingMode)
{
    switch (paddingMode)
    {
        case PaddingMode::Constant:   return arm_compute::PaddingMode::CONSTANT;
        case PaddingMode::Reflect:    return arm_compute::PaddingMode::REFLECT;
        case PaddingMode::Symmetric:  return arm_compute::PaddingMode::SYMMETRIC;
        default:                      throw InvalidArgumentException("Unsupported Padding Mode");
    }
}

inline arm_compute::ReductionOperation ConvertReductionOperationToAcl(const ReduceDescriptor& descriptor)
{
    switch (descriptor.m_ReduceOperation)
    {
        case ReduceOperation::Sum:    return arm_compute::ReductionOperation::SUM;
        case ReduceOperation::Mean:   return arm_compute::ReductionOperation::MEAN_SUM;
        case ReduceOperation::Max:    return arm_compute::ReductionOperation::MAX;
        case ReduceOperation::Min:    return arm_compute::ReductionOperation::MIN;
        case ReduceOperation::Prod:   return arm_compute::ReductionOperation::PROD;
        default:                      throw InvalidArgumentException("Unsupported Reduction operation");
    }
}

/// Function to compute the output tensor shape based on the axes and if keepDims is set.
inline const TensorInfo ComputeReductionTensorShape(const armnn::TensorInfo& input,
                                                    const std::vector<uint32_t>& vAxis,
                                                    const bool keepDims)
{
    auto reducedTensorInfo = input;
    unsigned int rank = reducedTensorInfo.GetNumDimensions();
    unsigned int outputRank = 0;
    // Calculate output dimension
    if (keepDims)
    {
        outputRank = rank;
    }
    else if (vAxis.empty())
    {
        outputRank = 1;
    }
    else if (vAxis.size() > reducedTensorInfo.GetNumDimensions())
    {
        throw LayerValidationException("ReduceLayer: Dimensions to reduce can not be bigger than input dimensions");
    }
    else
    {
        outputRank = reducedTensorInfo.GetNumDimensions() - armnn::numeric_cast<unsigned int>(vAxis.size());
        if (outputRank == 0)
        {
            outputRank = 1;
        }
    }
    std::vector<unsigned int> dimSizes(outputRank, 1);
    if (!vAxis.empty())
    {
        // Skip the dimension that has been reduced unless keepDims is true.
        unsigned int outputIndex = 0;
        for (unsigned int i = 0; i < reducedTensorInfo.GetNumDimensions(); ++i)
        {
            if (std::find(vAxis.begin(), vAxis.end(), i) == vAxis.end())
            {
                dimSizes[outputIndex] = armnn::numeric_cast<unsigned int>(reducedTensorInfo.GetShape()[i]);
                ++outputIndex;
            }
            else if (keepDims)
            {
                dimSizes[outputIndex] = 1;
                ++outputIndex;
            }
        }
    }
    const TensorShape inferredShape = TensorShape(outputRank, dimSizes.data());
    reducedTensorInfo.SetShape(inferredShape);
    return reducedTensorInfo;
}

/// Macro function check if layer with multiple axes is supported on each backend
#define IS_MULTI_AXES_REDUCE_SUPPORTED(func, input, desc, status)                 \
    armnn::TensorInfo inputTensorInfo = input;                                    \
    unsigned int recalulatedAxis = 0;                                             \
    std::vector<uint32_t> axes;                                                   \
                                                                                  \
    for (unsigned int i = 0; i != desc.m_vAxis.size(); ++i)                       \
    {                                                                             \
        axes.emplace_back(desc.m_vAxis[i]);                                       \
                                                                                  \
        const armnn::TensorInfo& reducedTensorInfo =                              \
            ComputeReductionTensorShape(input, axes, desc.m_KeepDims);            \
                                                                                  \
        std::vector<uint32_t> singleAxis(1, desc.m_vAxis[i] - recalulatedAxis);   \
                                                                                  \
        armnn::ReduceDescriptor newReduceDescriptor = desc;                       \
        newReduceDescriptor.m_vAxis.assign(singleAxis.begin(), singleAxis.end()); \
                                                                                  \
        status = func(inputTensorInfo, reducedTensorInfo, newReduceDescriptor);   \
        if (!status)                                                              \
        {                                                                         \
            break;                                                                \
        }                                                                         \
                                                                                  \
        if (!desc.m_KeepDims)                                                     \
        {                                                                         \
            recalulatedAxis++;                                                    \
        }                                                                         \
                                                                                  \
        inputTensorInfo = reducedTensorInfo;                                      \
    }

} // namespace armnn
