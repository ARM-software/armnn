//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClSoftmaxFloat32Workload.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"

namespace armnn
{

ClSoftmaxFloat32Workload::ClSoftmaxFloat32Workload(const SoftmaxQueueDescriptor& descriptor, const WorkloadInfo& info)
    : Float32Workload<SoftmaxQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClSoftmaxFloat32Workload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<ClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<ClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    m_SoftmaxLayer.configure(&input, &output, m_Data.m_Parameters.m_Beta);
}

void ClSoftmaxFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "ClSoftmaxFloat32Workload_Execute");
    m_SoftmaxLayer.run();
}

} //namespace armnn