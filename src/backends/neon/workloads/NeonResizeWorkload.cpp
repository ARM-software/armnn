//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonResizeWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <neon/NeonTensorHandle.hpp>

using namespace armnn::armcomputetensorutils;

namespace armnn
{

arm_compute::Status NeonResizeWorkloadValidate(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const ResizeDescriptor& descriptor)
{
    arm_compute::TensorInfo aclInputInfo  = BuildArmComputeTensorInfo(input);
    arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(descriptor.m_DataLayout);
    aclInputInfo.set_data_layout(aclDataLayout);
    aclOutputInfo.set_data_layout(aclDataLayout);

    arm_compute::InterpolationPolicy aclInterpolationPolicy =
            ConvertResizeMethodToAclInterpolationPolicy(descriptor.m_Method);

    return arm_compute::NEScale::validate(&aclInputInfo,
                                          &aclOutputInfo,
                                          aclInterpolationPolicy,
                                          arm_compute::BorderMode::REPLICATE,
                                          arm_compute::PixelValue(0.f),
                                          arm_compute::SamplingPolicy::TOP_LEFT);
}

NeonResizeWorkload::NeonResizeWorkload(const ResizeQueueDescriptor& descriptor,
                                                       const WorkloadInfo& info)
    : BaseWorkload<ResizeQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonResizeWorkload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    arm_compute::InterpolationPolicy aclInterpolationPolicy =
            ConvertResizeMethodToAclInterpolationPolicy(descriptor.m_Parameters.m_Method);

    m_ResizeLayer.configure(&input,
                            &output,
                            aclInterpolationPolicy,
                            arm_compute::BorderMode::REPLICATE,
                            arm_compute::PixelValue(0.f),
                            arm_compute::SamplingPolicy::TOP_LEFT);
};

void NeonResizeWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonResizeWorkload_Execute");
    m_ResizeLayer.run();
}

} //namespace armnn
