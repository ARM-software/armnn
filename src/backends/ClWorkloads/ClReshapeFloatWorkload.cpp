//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClReshapeFloatWorkload.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"

#include "ClWorkloadUtils.hpp"

namespace armnn
{

ClReshapeFloatWorkload::ClReshapeFloatWorkload(const ReshapeQueueDescriptor& descriptor, const WorkloadInfo& info)
    : FloatWorkload<ReshapeQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClReshapeFloatWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_Layer.configure(&input, &output);
}

void ClReshapeFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClReshapeFloatWorkload_Execute");
    m_Layer.run();
}

} //namespace armnn

