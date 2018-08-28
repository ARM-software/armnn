//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonDepthwiseConvolutionBaseWorkload.hpp"

#include "backends/ArmComputeTensorUtils.hpp"

namespace armnn
{

arm_compute::Status NeonDepthwiseConvolutionWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const DepthwiseConvolution2dDescriptor& descriptor,
    const TensorInfo& weights,
    const boost::optional<TensorInfo>& biases)
{
    const arm_compute::TensorInfo aclInputInfo =
        armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo =
        armcomputetensorutils::BuildArmComputeTensorInfo(output);
    const arm_compute::TensorInfo aclWeightsInfo =
        armcomputetensorutils::BuildArmComputeTensorInfo(weights);

    arm_compute::TensorInfo aclBiasesInfo;
    arm_compute::TensorInfo *optionalAclBiasesInfo = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        BOOST_ASSERT(biases.is_initialized());

        aclBiasesInfo = armcomputetensorutils::BuildArmComputeTensorInfo(biases.get());
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