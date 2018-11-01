//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClConvertFp32ToFp16Workload.hpp"
#include <cl/ClTensorHandle.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

static constexpr arm_compute::ConvertPolicy g_AclConvertPolicy = arm_compute::ConvertPolicy::SATURATE;

ClConvertFp32ToFp16Workload::ClConvertFp32ToFp16Workload(
    const ConvertFp32ToFp16QueueDescriptor& descriptor, const WorkloadInfo& info) :
    Float32ToFloat16Workload<ConvertFp32ToFp16QueueDescriptor>(descriptor, info)
{
    this->m_Data.ValidateInputsOutputs("ClConvertFp32ToFp16Workload", 1, 1);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(this->m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(this->m_Data.m_Outputs[0])->GetTensor();

    m_Layer.configure(&input, &output, g_AclConvertPolicy, 0);
}

void ClConvertFp32ToFp16Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClConvertFp32ToFp16Workload_Execute");
    RunClFunction(m_Layer, CHECK_LOCATION());
}

arm_compute::Status ClConvertFp32ToFp16WorkloadValidate(const TensorInfo& input, const TensorInfo& output)
{
    if (input.GetDataType() != DataType::Float32)
    {
        return arm_compute::Status(arm_compute::ErrorCode::RUNTIME_ERROR, "Input should be Float32");
    }
    if (output.GetDataType() != DataType::Float16)
    {
        return arm_compute::Status(arm_compute::ErrorCode::RUNTIME_ERROR, "Output should be Float16");
    }

    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    const arm_compute::Status aclStatus = arm_compute::CLDepthConvertLayer::validate(
        &aclInputInfo, &aclOutputInfo, g_AclConvertPolicy, 0);

    return aclStatus;
}


} //namespace armnn
