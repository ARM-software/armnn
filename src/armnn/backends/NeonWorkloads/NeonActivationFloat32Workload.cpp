//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonActivationFloat32Workload.hpp"
#include "backends/ArmComputeUtils.hpp"


namespace armnn
{
NeonActivationFloat32Workload::NeonActivationFloat32Workload(const ActivationQueueDescriptor& descriptor,
                                                             const WorkloadInfo&              info)
    : Float32Workload<ActivationQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonActivationFloat32Workload", 1, 1);

    const arm_compute::ActivationLayerInfo activationLayerInfo =
        ConvertActivationDescriptorToAclActivationLayerInfo(m_Data.m_Parameters);

    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_ActivationLayer.configure(&input, &output, activationLayerInfo);
}

void NeonActivationFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "NeonActivationFloat32Workload_Execute");
    m_ActivationLayer.run();
}

} //namespace armnn

