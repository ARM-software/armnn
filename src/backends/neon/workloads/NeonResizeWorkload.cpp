//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonResizeWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>

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

    arm_compute::SamplingPolicy samplingPolicy = descriptor.m_HalfPixelCenters ? arm_compute::SamplingPolicy::CENTER :
                                                                                 arm_compute::SamplingPolicy::TOP_LEFT;

    bool usePadding = false;

    return arm_compute::NEScale::validate(&aclInputInfo,
                                          &aclOutputInfo,
                                          arm_compute::ScaleKernelInfo(aclInterpolationPolicy,
                                                                       arm_compute::BorderMode::REPLICATE,
                                                                       arm_compute::PixelValue(0.f),
                                                                       samplingPolicy,
                                                                       usePadding,
                                                                       descriptor.m_AlignCorners));

}

NeonResizeWorkload::NeonResizeWorkload(const ResizeQueueDescriptor& descriptor,
                                                       const WorkloadInfo& info)
    : NeonBaseWorkload<ResizeQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonResizeWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("NeonResizeWorkload", 1, 1);

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    arm_compute::InterpolationPolicy aclInterpolationPolicy =
            ConvertResizeMethodToAclInterpolationPolicy(descriptor.m_Parameters.m_Method);

    arm_compute::SamplingPolicy samplingPolicy = descriptor.m_Parameters.m_HalfPixelCenters
                                                 ? arm_compute::SamplingPolicy::CENTER
                                                 : arm_compute::SamplingPolicy::TOP_LEFT;

    bool usePadding = false;

    m_ResizeLayer.configure(&input,
                            &output,
                            arm_compute::ScaleKernelInfo(aclInterpolationPolicy,
                                                         arm_compute::BorderMode::REPLICATE,
                                                         arm_compute::PixelValue(0.f),
                                                         samplingPolicy,
                                                         usePadding,
                                                         descriptor.m_Parameters.m_AlignCorners));
};

void NeonResizeWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonResizeWorkload_Execute", this->GetGuid());
    m_ResizeLayer.run();
}

} //namespace armnn
