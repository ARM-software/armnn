//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClActivationFloat32Workload.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/ArmComputeUtils.hpp"

namespace armnn
{

ClActivationFloat32Workload::ClActivationFloat32Workload(const ActivationQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info)
    : Float32Workload<ActivationQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClActivationFloat32Workload", 1, 1);

    const arm_compute::ActivationLayerInfo activationLayerInfo =
        ConvertActivationDescriptorToAclActivationLayerInfo(m_Data.m_Parameters);

    arm_compute::ICLTensor& input  = static_cast<ClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<ClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    m_ActivationLayer.configure(&input, &output, activationLayerInfo);
}

void ClActivationFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "ClActivationFloat32Workload_Execute");
    m_ActivationLayer.run();
}

} //namespace armnn