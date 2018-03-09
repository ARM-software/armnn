//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClReshapeFloat32Workload.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"

namespace armnn
{

ClReshapeFloat32Workload::ClReshapeFloat32Workload(const ReshapeQueueDescriptor& descriptor, const WorkloadInfo& info)
    : Float32Workload<ReshapeQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClReshapeFloat32Workload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_Layer.configure(&input, &output);
}

void ClReshapeFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "ClReshapeFloat32Workload_Execute");
    m_Layer.run();
}

} //namespace armnn

