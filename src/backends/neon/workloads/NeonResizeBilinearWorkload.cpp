//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonResizeBilinearWorkload.hpp"

#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <neon/NeonTensorHandle.hpp>
#include <neon/NeonLayerSupport.hpp>

using namespace armnn::armcomputetensorutils;

namespace armnn
{

arm_compute::Status NeonResizeBilinearWorkloadValidate(const TensorInfo& input, const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInputInfo  = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::NEScale::validate(&aclInputInfo,
                                          &aclOutputInfo,
                                          arm_compute::InterpolationPolicy::BILINEAR,
                                          arm_compute::BorderMode::REPLICATE,
                                          arm_compute::PixelValue(0.f),
                                          arm_compute::SamplingPolicy::TOP_LEFT);
}

NeonResizeBilinearWorkload::NeonResizeBilinearWorkload(const ResizeBilinearQueueDescriptor& descriptor,
                                                       const WorkloadInfo& info)
    : BaseWorkload<ResizeBilinearQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonResizeBilinearWorkload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    m_ResizeBilinearLayer.configure(&input,
                                    &output,
                                    arm_compute::InterpolationPolicy::BILINEAR,
                                    arm_compute::BorderMode::REPLICATE,
                                    arm_compute::PixelValue(0.f),
                                    arm_compute::SamplingPolicy::TOP_LEFT);
};

void NeonResizeBilinearWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonResizeBilinearWorkload_Execute");
    m_ResizeBilinearLayer.run();
}

} //namespace armnn
