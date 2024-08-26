//
// Copyright © 2021-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClCastWorkload.hpp"
#include "ClWorkloadUtils.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <cl/ClTensorHandle.hpp>

namespace armnn
{

static constexpr arm_compute::ConvertPolicy g_AclConvertPolicy = arm_compute::ConvertPolicy::SATURATE;

arm_compute::Status ClCastValidate(const TensorInfo& input, const TensorInfo& output)
{
    // ACL doesn't have a Boolean type - the armnn Boolean is evaluated as an ACL U8.
    // This causes issues when casting numbers to Boolean, as casting float to U8 truncates decimal points
    // and casting negative signed ints to U8 clamps to 0, but a cast to Boolean returns true for anything non-zero.
    // For example, float to U8 expects 0.1f -> 0u, but float to Boolean 0.1f -> true.
    // ACL isn't aware of the Boolean type, so this check has to be here.
    if(output.GetDataType() == armnn::DataType::Boolean)
    {
        return arm_compute::Status{arm_compute::ErrorCode::RUNTIME_ERROR, "Cast to Boolean unsupported"};
    }

    const arm_compute::TensorInfo aclInput  = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::CLCast::validate(&aclInput, &aclOutput, g_AclConvertPolicy);
}

ClCastWorkload::ClCastWorkload(const CastQueueDescriptor& descriptor,
                               const WorkloadInfo& info,
                               const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<CastQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClCastWorkload", 1, 1);

    arm_compute::ICLTensor& input  = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    {
        ARMNN_SCOPED_PROFILING_EVENT_CL_NAME_GUID("ClCastWorkload_configure");
        m_CastLayer.configure(clCompileContext, &input, &output, g_AclConvertPolicy);
    }
}

void ClCastWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_NAME_GUID("ClCastWorkload_Execute");
    RunClFunction(m_CastLayer, CHECK_LOCATION());
}

} // namespace armnn
