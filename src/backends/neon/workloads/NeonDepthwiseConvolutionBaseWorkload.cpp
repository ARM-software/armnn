//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonDepthwiseConvolutionBaseWorkload.hpp"

#include <backends/aclCommon/ArmComputeTensorUtils.hpp>

namespace armnn
{

arm_compute::Status NeonDepthwiseConvolutionWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const DepthwiseConvolution2dDescriptor& descriptor,
    const TensorInfo& weights,
    const Optional<TensorInfo>& biases)
{
    const arm_compute::TensorInfo aclInputInfo =
        armcomputetensorutils::BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo =
        armcomputetensorutils::BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclWeightsInfo =
        armcomputetensorutils::BuildArmComputeTensorInfo(weights, descriptor.m_DataLayout);

    arm_compute::TensorInfo aclBiasesInfo;
    arm_compute::TensorInfo *optionalAclBiasesInfo = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        BOOST_ASSERT(biases.has_value());

        aclBiasesInfo = armcomputetensorutils::BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        optionalAclBiasesInfo = &aclBiasesInfo;
    }

    const arm_compute::PadStrideInfo aclPadStrideInfo =
        armcomputetensorutils::BuildArmComputePadStrideInfo(descriptor);
    const unsigned int aclDepthMultiplier = weights.GetShape()[0];

    return arm_compute::NEDepthwiseConvolutionLayer::validate(&aclInputInfo,
                                                              &aclWeightsInfo,
                                                              optionalAclBiasesInfo,
                                                              &aclOutputInfo,
                                                              aclPadStrideInfo,
                                                              aclDepthMultiplier);
}

}