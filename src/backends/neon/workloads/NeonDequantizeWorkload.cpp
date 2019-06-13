//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonDequantizeWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <neon/NeonTensorHandle.hpp>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status NeonDequantizeWorkloadValidate(const TensorInfo& input,
                                                   const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = BuildArmComputeTensorInfo(output);

    return arm_compute::NEDequantizationLayer::validate(&aclInput, &aclOutput);
}

NeonDequantizeWorkload::NeonDequantizeWorkload(const DequantizeQueueDescriptor& descriptor, const WorkloadInfo& info)
        : BaseWorkload<DequantizeQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonDequantizeWorkload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_Layer.reset(new arm_compute::NEDequantizationLayer());
    m_Layer->configure(&input, &output);
    m_Layer->prepare();
}

void NeonDequantizeWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonDequantizeWorkload_Execute");
    m_Layer->run();
}

} //namespace armnn

