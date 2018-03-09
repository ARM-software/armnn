//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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

    std::string reasonIfUnsupported;
    if (!IsNeonActivationUint8Supported(&reasonIfUnsupported, m_Data.m_Parameters))
    {
        throw InvalidArgumentException(reasonIfUnsupported);
    }

    // Only BoundedReLu is supported (see IsNeonActivationUint8Supported)
    arm_compute::ActivationLayerInfo layerInfo(arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                               m_Data.m_Parameters.m_A,
                                               m_Data.m_Parameters.m_B);

    m_Data.ValidateInputsOutputs("NeonActivationUint8Workload", 1, 1);

    arm_compute::ITensor& input  = static_cast<NeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = static_cast<NeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_ActivationLayer.configure(&input, &output, layerInfo);
}

void NeonActivationUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "NeonActivationUint8Workload_Execute");

    m_ActivationLayer.run();
}
} //namespace armnn
