//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClReshapeWorkload.hpp"
#include <cl/ClTensorHandle.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{

arm_compute::Status ClReshapeWorkloadValidate(const TensorInfo& input,
                                              const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::CLReshapeLayer::validate(&aclInputInfo, &aclOutputInfo);
}

ClReshapeWorkload::ClReshapeWorkload(const ReshapeQueueDescriptor& descriptor, const WorkloadInfo& info)
    : BaseWorkload<ReshapeQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClReshapeWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_Layer.configure(&input, &output);
}

void ClReshapeWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClReshapeWorkload_Execute");
    RunClFunction(m_Layer, CHECK_LOCATION());
}

} //namespace armnn
