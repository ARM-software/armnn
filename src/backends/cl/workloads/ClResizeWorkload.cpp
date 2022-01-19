//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClResizeWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <armnn/backends/TensorHandle.hpp>

#include <cl/ClTensorHandle.hpp>

using namespace armnn::armcomputetensorutils;

namespace armnn
{

arm_compute::Status ClResizeWorkloadValidate(const TensorInfo& input,
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

    return arm_compute::CLScale::validate(&aclInputInfo,
                                          &aclOutputInfo,
                                          arm_compute::ScaleKernelInfo(aclInterpolationPolicy,
                                                                       arm_compute::BorderMode::REPLICATE,
                                                                       arm_compute::PixelValue(0.f),
                                                                       samplingPolicy,
                                                                       true,
                                                                       descriptor.m_AlignCorners));
}

ClResizeWorkload::ClResizeWorkload(const ResizeQueueDescriptor& descriptor,
                                   const WorkloadInfo& info,
                                   const arm_compute::CLCompileContext& clCompileContext)
  : ClBaseWorkload<ResizeQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClResizeWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClResizeWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    arm_compute::InterpolationPolicy aclInterpolationPolicy =
        ConvertResizeMethodToAclInterpolationPolicy(descriptor.m_Parameters.m_Method);

    arm_compute::SamplingPolicy samplingPolicy = descriptor.m_Parameters.m_HalfPixelCenters
                                                 ? arm_compute::SamplingPolicy::CENTER
                                                 : arm_compute::SamplingPolicy::TOP_LEFT;

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClResizeWorkload_configure");
        m_ResizeLayer.configure(clCompileContext,
                                &input,
                                &output,
                                arm_compute::ScaleKernelInfo(aclInterpolationPolicy,
                                                             arm_compute::BorderMode::REPLICATE,
                                                             arm_compute::PixelValue(0.f),
                                                             samplingPolicy,
                                                             true,
                                                             descriptor.m_Parameters.m_AlignCorners));
    }

};

void ClResizeWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClResizeWorkload_Execute", this->GetGuid());
    RunClFunction(m_ResizeLayer, CHECK_LOCATION());
}

} //namespace armnn
