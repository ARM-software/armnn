//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "UtilsGpuFsa.hpp"
#include "aclCommon/ArmComputeTensorUtils.hpp"
#include "aclCommon/ArmComputeUtils.hpp"

using namespace armnn;
using namespace armnn::armcomputetensorutils;
using namespace arm_compute::experimental::dynamic_fusion;

Conv2dAttributes CreateConv2dAttributes(const Convolution2dDescriptor& descriptor)
{
    const arm_compute::Padding2D padInfo      = BuildArmComputePaddingInfo(descriptor);
    const arm_compute::Size2D    strideInfo   = BuildArmComputeSize2D(descriptor.m_StrideX, descriptor.m_StrideY);
    const arm_compute::Size2D    dilationInfo = BuildArmComputeSize2D(descriptor.m_DilationX, descriptor.m_DilationY);

    arm_compute::experimental::dynamic_fusion::Conv2dAttributes conv2dAttributes{};
    conv2dAttributes.pad(padInfo);
    conv2dAttributes.stride(strideInfo);
    conv2dAttributes.dilation(dilationInfo);

    return conv2dAttributes;
}

arm_compute::experimental::dynamic_fusion::DepthwiseConv2dAttributes
CreateDWConv2dAttributes(const DepthwiseConvolution2dDescriptor& descriptor, const unsigned int aclDepthMultiplier)
{
    const arm_compute::Padding2D padInfo      = BuildArmComputePaddingInfo(descriptor);
    const arm_compute::Size2D    strideInfo   = BuildArmComputeSize2D(descriptor.m_StrideX, descriptor.m_StrideY);
    const arm_compute::Size2D    dilationInfo = BuildArmComputeSize2D(descriptor.m_DilationX, descriptor.m_DilationY);

    arm_compute::experimental::dynamic_fusion::DepthwiseConv2dAttributes depthwiseConv2dAttributes{};
    depthwiseConv2dAttributes.pad(padInfo);
    depthwiseConv2dAttributes.stride(strideInfo);
    depthwiseConv2dAttributes.dilation(dilationInfo);
    depthwiseConv2dAttributes.depth_multiplier(aclDepthMultiplier);

    return depthwiseConv2dAttributes;
}

arm_compute::experimental::dynamic_fusion::Pool2dAttributes
CreatePool2dAttributes(const Pooling2dDescriptor& descriptor)
{
    const arm_compute::PoolingType poolType = ConvertPoolingAlgorithmToAclPoolingType(descriptor.m_PoolType);
    const arm_compute::Padding2D   padding  = BuildArmComputePaddingInfo(descriptor);
    const arm_compute::Size2D      poolSize = BuildArmComputeSize2D(descriptor.m_PoolWidth, descriptor.m_PoolHeight);
    const arm_compute::Size2D      strides  = BuildArmComputeSize2D(descriptor.m_StrideX, descriptor.m_StrideY);
    const bool excludePadding = (descriptor.m_PaddingMethod == PaddingMethod::Exclude);

    arm_compute::experimental::dynamic_fusion::Pool2dAttributes pool2dAttributes{};
    pool2dAttributes.pool_type(poolType);
    pool2dAttributes.pad(padding);
    pool2dAttributes.pool_size(poolSize);
    pool2dAttributes.stride(strides);
    pool2dAttributes.exclude_padding(excludePadding);

    return pool2dAttributes;
}

arm_compute::experimental::dynamic_fusion::ResizeAttributes
CreateResizeAttributes(const armnn::ResizeDescriptor& descriptor)
{
    arm_compute::experimental::dynamic_fusion::ResizeAttributes resizeAttributes{};
    resizeAttributes.output_width(static_cast<int32_t>(descriptor.m_TargetWidth));
    resizeAttributes.output_height(static_cast<int32_t>(descriptor.m_TargetHeight));
    resizeAttributes.interpolation_policy(descriptor.m_Method == ResizeMethod::Bilinear ?
                                          arm_compute::InterpolationPolicy::BILINEAR :
                                          arm_compute::InterpolationPolicy::NEAREST_NEIGHBOR);
    resizeAttributes.sampling_policy(descriptor.m_HalfPixelCenters ? arm_compute::SamplingPolicy::CENTER
                                                                   : arm_compute::SamplingPolicy::TOP_LEFT);
    resizeAttributes.align_corners(descriptor.m_AlignCorners);

    return resizeAttributes;
}