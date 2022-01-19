//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClInstanceNormalizationWorkload.hpp"
#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <cl/ClTensorHandle.hpp>

using namespace armnn::armcomputetensorutils;

namespace armnn
{

arm_compute::Status ClInstanceNormalizationWorkloadValidate(const TensorInfo& input,
                                                            const TensorInfo& output,
                                                            const InstanceNormalizationDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo  = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);

    return arm_compute::CLInstanceNormalizationLayer::validate(&aclInputInfo,
                                                               &aclOutputInfo,
                                                               descriptor.m_Gamma,
                                                               descriptor.m_Beta,
                                                               descriptor.m_Eps);
}

ClInstanceNormalizationWorkload::ClInstanceNormalizationWorkload(
    const InstanceNormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info,
    const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<InstanceNormalizationQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClInstanceNormalizationWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClInstanceNormalizationWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClInstanceNormalizationWorkload_configure");
        m_Layer.configure(clCompileContext,
                          &input,
                          &output,
                          descriptor.m_Parameters.m_Gamma,
                          descriptor.m_Parameters.m_Beta,
                          descriptor.m_Parameters.m_Eps);
    }
};

void ClInstanceNormalizationWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClInstanceNormalizationWorkload_Execute", this->GetGuid());
    RunClFunction(m_Layer, CHECK_LOCATION());
}

} // namespace armnn
