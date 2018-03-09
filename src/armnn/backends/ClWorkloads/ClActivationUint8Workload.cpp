//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClActivationUint8Workload.hpp"
#include "backends/ClLayerSupport.hpp"

#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"
namespace armnn
{

ClActivationUint8Workload::ClActivationUint8Workload(const ActivationQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info)
    : Uint8Workload<ActivationQueueDescriptor>(descriptor, info)
{

    std::string reasonIfUnsupported;
    if (!IsClActivationUint8Supported(&reasonIfUnsupported, m_Data.m_Parameters))
    {
        throw InvalidArgumentException(reasonIfUnsupported);
    }

    // Only BoundedReLu is supported (see IsClActivationUint8Supported)
    arm_compute::ActivationLayerInfo layerInfo(arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                               m_Data.m_Parameters.m_A,
                                               m_Data.m_Parameters.m_B);

    m_Data.ValidateInputsOutputs("ClActivationUint8Workload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<ClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<ClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_ActivationLayer.configure(&input, &output, layerInfo);
}

void ClActivationUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "ClActivationUint8Workload_Execute");

    m_ActivationLayer.run();
}

} //namespace Armnn


