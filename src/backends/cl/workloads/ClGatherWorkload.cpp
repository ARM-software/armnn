//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClGatherWorkload.hpp"
#include "ClWorkloadUtils.hpp"
#include <aclCommon/ArmComputeUtils.hpp>
#include <cl/ClTensorHandle.hpp>

using namespace armnn::armcomputetensorutils;

namespace armnn
{
arm_compute::Status ClGatherWorkloadValidate(const TensorInfo& input,
                                             const TensorInfo& indices,
                                             const TensorInfo& output,
                                             const GatherDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInput   = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclIndices = BuildArmComputeTensorInfo(indices);
    const arm_compute::TensorInfo aclOutput  = BuildArmComputeTensorInfo(output);

    int aclAxis = ComputeAclAxis(descriptor.m_Axis, input);

    return arm_compute::CLGather::validate(&aclInput, &aclIndices, &aclOutput, aclAxis);
}

ClGatherWorkload::ClGatherWorkload(const GatherQueueDescriptor& descriptor,
                                   const WorkloadInfo& info)
        : BaseWorkload<GatherQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClGatherWorkload", 1, 1);

    arm_compute::ICLTensor& input    = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& indices  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& output   = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    int aclAxis = ComputeAclAxis(descriptor.m_Parameters.m_Axis, info.m_InputTensorInfos[0]);

    m_Layer.configure(&input, &indices, &output, aclAxis);
};

void ClGatherWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClGatherWorkload_Execute");
    RunClFunction(m_Layer, CHECK_LOCATION());
}
} // namespace armnn
