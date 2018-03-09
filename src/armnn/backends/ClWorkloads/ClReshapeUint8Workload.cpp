//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClReshapeUint8Workload.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"

namespace armnn
{
ClReshapeUint8Workload::ClReshapeUint8Workload(const ReshapeQueueDescriptor& descriptor, const WorkloadInfo& info)
    : Uint8Workload<ReshapeQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClReshapeUint8Workload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    m_Layer.configure(&input, &output);
}

void ClReshapeUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "ClReshapeUint8Workload_Execute");

    m_Layer.run();
}

} //namespace armnn