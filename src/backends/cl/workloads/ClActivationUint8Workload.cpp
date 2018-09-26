//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClActivationUint8Workload.hpp"
#include <backends/cl/ClLayerSupport.hpp>

#include <backends/aclCommon/ArmComputeUtils.hpp>
#include <backends/cl/ClTensorHandle.hpp>
#include <backends/CpuTensorHandle.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{

ClActivationUint8Workload::ClActivationUint8Workload(const ActivationQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info)
    : Uint8Workload<ActivationQueueDescriptor>(descriptor, info)
{
    auto activation = ConvertActivationFunctionToAclActivationFunction(m_Data.m_Parameters.m_Function);
    arm_compute::ActivationLayerInfo layerInfo(activation,
                                               m_Data.m_Parameters.m_A,
                                               m_Data.m_Parameters.m_B);

    m_Data.ValidateInputsOutputs("ClActivationUint8Workload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<ClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<ClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_ActivationLayer.configure(&input, &output, layerInfo);
}

void ClActivationUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClActivationUint8Workload_Execute");

    m_ActivationLayer.run();
}

} //namespace Armnn


