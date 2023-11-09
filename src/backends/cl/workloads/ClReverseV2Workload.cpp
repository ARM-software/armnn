//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClReverseV2Workload.hpp"
#include "ClWorkloadUtils.hpp"
#include <aclCommon/ArmComputeUtils.hpp>
#include <cl/ClTensorHandle.hpp>
#include <backendsCommon/WorkloadUtils.hpp>

using namespace armnn::armcomputetensorutils;

namespace armnn
{
arm_compute::Status ClReverseV2WorkloadValidate(const TensorInfo& input,
                                                const TensorInfo& axis,
                                                const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclAxis = BuildArmComputeTensorInfo(axis);
    const arm_compute::TensorInfo aclOutput = BuildArmComputeTensorInfo(output);

    return arm_compute::CLReverse::validate(&aclInput, &aclOutput, &aclAxis, true);
}

ClReverseV2Workload::ClReverseV2Workload(const armnn::ReverseV2QueueDescriptor &descriptor,
                                         const armnn::WorkloadInfo &info,
                                         const arm_compute::CLCompileContext& clCompileContext)
        : BaseWorkload<ReverseV2QueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClReverseV2Workload", 2, 1);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& axis = static_cast<IClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    {
        ARMNN_SCOPED_PROFILING_EVENT_CL_NAME_GUID("ClReverseV2Workload_configure");
        m_Layer.configure(clCompileContext, &input, &output, &axis, true);
    }
}

void ClReverseV2Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_NAME_GUID("ClReverseV2Workload_Execute");
    m_Layer.run();
}

} //namespace armnn