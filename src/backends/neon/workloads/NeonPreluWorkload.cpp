//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonPreluWorkload.hpp"
#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <arm_compute/runtime/NEON/functions/NEPReluLayer.h>

namespace armnn
{

arm_compute::Status NeonPreluWorkloadValidate(const TensorInfo& input,
                                              const TensorInfo& alpha,
                                              const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclAlpha = armcomputetensorutils::BuildArmComputeTensorInfo(alpha);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::NEPReluLayer::validate(&aclInput,
                                               &aclAlpha,
                                               &aclOutput);
}

NeonPreluWorkload::NeonPreluWorkload(const PreluQueueDescriptor& descriptor,
                                     const WorkloadInfo& info)
        : NeonBaseWorkload<PreluQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonPreluWorkload", 1, 1);

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& alpha = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    auto layer = std::make_unique<arm_compute::NEPReluLayer>();
    layer->configure(&input, &alpha, &output);

    m_PreluLayer.reset(layer.release());
}

void NeonPreluWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonPreluWorkload_Execute", this->GetGuid());
    m_PreluLayer->run();
}

} //namespace armnn