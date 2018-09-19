//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonActivationUint8Workload.hpp"
#include "backends/ArmComputeUtils.hpp"
#include "backends/NeonLayerSupport.hpp"

namespace armnn
{
NeonActivationUint8Workload::NeonActivationUint8Workload(const ActivationQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info)
    : Uint8Workload<ActivationQueueDescriptor>(descriptor, info)
{
    auto activation = ConvertActivationFunctionToAclActivationFunction(m_Data.m_Parameters.m_Function);
    arm_compute::ActivationLayerInfo layerInfo(activation,
                                               m_Data.m_Parameters.m_A,
                                               m_Data.m_Parameters.m_B);

    m_Data.ValidateInputsOutputs("NeonActivationUint8Workload", 1, 1);

    arm_compute::ITensor& input  = static_cast<NeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = static_cast<NeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_ActivationLayer.configure(&input, &output, layerInfo);
}

void NeonActivationUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonActivationUint8Workload_Execute");

    m_ActivationLayer.run();
}
} //namespace armnn
