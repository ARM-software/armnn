//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#if ARMCOMPUTENEON_ENABLED || ARMCOMPUTECL_ENABLED

#include <armnn/Tensor.hpp>
#include <armnn/Descriptors.hpp>

#include <arm_compute/core/Types.h>

namespace armnn
{

inline arm_compute::NormalizationLayerInfo
CreateAclNormalizationLayerInfoForL2Normalization(const armnn::TensorInfo& tensorInfo)
{
    const unsigned int depth = tensorInfo.GetShape()[1];

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
        default:                                throw InvalidArgumentException("Unsupported activation function");
    }
}

inline arm_compute::ActivationLayerInfo
ConvertActivationDescriptorToAclActivationLayerInfo(const ActivationDescriptor& actDesc)
{
    return arm_compute::ActivationLayerInfo(ConvertActivationFunctionToAclActivationFunction(actDesc.m_Function),
        actDesc.m_A, actDesc.m_B);
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
ConvertFullyConnectedDescriptorToAclFullyConnectedLayerInfo(const FullyConnectedDescriptor& fullyConnectedDesc)
{
    arm_compute::FullyConnectedLayerInfo fc_info;
    fc_info.transpose_weights = fullyConnectedDesc.m_TransposeWeightMatrix;
    return fc_info;
}

}

#endif // ARMCOMPUTENEON_ENABLED || ARMCOMPUTECL_ENABLED
