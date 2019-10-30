//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClFloorFloatWorkload.hpp"
#include <cl/ClTensorHandle.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{

arm_compute::Status ClFloorWorkloadValidate(const TensorInfo& input,
                                            const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput  = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::CLFloor::validate(&aclInput, &aclOutput);
}

ClFloorFloatWorkload::ClFloorFloatWorkload(const FloorQueueDescriptor& descriptor, const WorkloadInfo& info)
    : FloatWorkload<FloorQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClFloorFloatWorkload", 1, 1);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_Layer.configure(&input, &output);
}

void ClFloorFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClFloorFloatWorkload_Execute");
    RunClFunction(m_Layer, CHECK_LOCATION());
}

} //namespace armnn
