//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonCastWorkload.hpp"
#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>

namespace armnn
{

static constexpr arm_compute::ConvertPolicy g_AclConvertPolicy = arm_compute::ConvertPolicy::SATURATE;

arm_compute::Status NeonCastValidate(const TensorInfo& input, const TensorInfo& output)
{
    arm_compute::TensorInfo aclInput  = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::NECast::validate(&aclInput, &aclOutput, g_AclConvertPolicy);
}

NeonCastWorkload::NeonCastWorkload(const CastQueueDescriptor& descriptor, const WorkloadInfo& info)
    : NeonBaseWorkload<CastQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonCastWorkload", 1, 1);

    arm_compute::ITensor& input  = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_CastLayer.configure(&input, &output, g_AclConvertPolicy);
}

void NeonCastWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonCastWorkload_Execute", this->GetGuid());
    m_CastLayer.run();
}

} // namespace armnn
