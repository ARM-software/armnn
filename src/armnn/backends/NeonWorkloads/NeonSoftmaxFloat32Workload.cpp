//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonSoftmaxFloat32Workload.hpp"

namespace armnn
{
NeonSoftmaxFloat32Workload::NeonSoftmaxFloat32Workload(const SoftmaxQueueDescriptor& descriptor,
                                                       const WorkloadInfo& info)
    : Float32Workload<SoftmaxQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonSoftmaxFloat32Workload", 1, 1);

    // The ArmCompute softmax layer uses 2D input/output tensors, so flatten the first three dimensions
    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_SoftmaxLayer.configure(&input, &output, m_Data.m_Parameters.m_Beta);
}

void NeonSoftmaxFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "NeonSoftmaxFloat32Workload_Execute");
    m_SoftmaxLayer.run();
}
} //namespace armnn



