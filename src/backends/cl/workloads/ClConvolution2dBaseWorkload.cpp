//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClConvolution2dBaseWorkload.hpp"
#include <backends/cl/ClLayerSupport.hpp>
#include <backends/cl/ClTensorHandle.hpp>
#include <backends/aclCommon/ArmComputeUtils.hpp>
#include <backends/aclCommon/ArmComputeTensorUtils.hpp>

#include <arm_compute/runtime/CL/functions/CLConvolutionLayer.h>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClConvolution2dWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const Convolution2dDescriptor& descriptor,
    const TensorInfo& weights,
    const boost::optional<TensorInfo>& biases)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclWeightsInfo = BuildArmComputeTensorInfo(weights, descriptor.m_DataLayout);

    arm_compute::TensorInfo aclBiasesInfo;
    arm_compute::TensorInfo *optionalAclBiasesInfo = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        BOOST_ASSERT(biases.is_initialized());

        aclBiasesInfo = BuildArmComputeTensorInfo(biases.get(), descriptor.m_DataLayout);
        optionalAclBiasesInfo = &aclBiasesInfo;
    }

    arm_compute::PadStrideInfo layerInfo = BuildArmComputePadStrideInfo(descriptor);

    return arm_compute::CLConvolutionLayer::validate(&aclInputInfo,
                                                     &aclWeightsInfo,
                                                     optionalAclBiasesInfo,
                                                     &aclOutputInfo,
                                                     layerInfo);
}

}
