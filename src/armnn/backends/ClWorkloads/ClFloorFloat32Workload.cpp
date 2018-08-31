//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClFloorFloat32Workload.hpp"
#include "backends/ClTensorHandle.hpp"

namespace armnn
{

ClFloorFloat32Workload::ClFloorFloat32Workload(const FloorQueueDescriptor& descriptor, const WorkloadInfo& info)
    : FloatWorkload<FloorQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClFloorFloat32Workload", 1, 1);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_Layer.configure(&input, &output);
}

void ClFloorFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClFloorFloat32Workload_Execute");
    m_Layer.run();
}

} //namespace armnn
