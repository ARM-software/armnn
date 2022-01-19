//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonReshapeWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <arm_compute/runtime/NEON/functions/NEReshapeLayer.h>

namespace armnn
{

arm_compute::Status NeonReshapeWorkloadValidate(const TensorInfo& input,
                                                const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::NEReshapeLayer::validate(&aclInputInfo, &aclOutputInfo);
}

NeonReshapeWorkload::NeonReshapeWorkload(const ReshapeQueueDescriptor& descriptor,
                                         const WorkloadInfo& info)
    : NeonBaseWorkload<ReshapeQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonReshapeWorkload", 1, 1);

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    auto layer = std::make_unique<arm_compute::NEReshapeLayer>();
    layer->configure(&input, &output);
    m_Layer.reset(layer.release());
}

void NeonReshapeWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonReshapeWorkload_Execute", this->GetGuid());
    m_Layer->run();
}

} //namespace armnn
