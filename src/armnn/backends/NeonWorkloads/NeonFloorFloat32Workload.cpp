//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonFloorFloat32Workload.hpp"

namespace armnn
{
NeonFloorFloat32Workload::NeonFloorFloat32Workload(const FloorQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info)
    : Float32Workload<FloorQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonFloorFloat32Workload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_Layer.configure(&input, &output);
}

void NeonFloorFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "NeonFloorFloat32Workload_Execute");
    m_Layer.run();
}
} //namespace armnn



