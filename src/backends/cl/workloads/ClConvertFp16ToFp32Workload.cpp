//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClConvertFp16ToFp32Workload.hpp"
#include <cl/ClTensorHandle.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

static constexpr arm_compute::ConvertPolicy g_AclConvertPolicy = arm_compute::ConvertPolicy::SATURATE;

ClConvertFp16ToFp32Workload::ClConvertFp16ToFp32Workload(
    const ConvertFp16ToFp32QueueDescriptor& descriptor,
    const WorkloadInfo& info,
    const arm_compute::CLCompileContext& clCompileContext) :
    Float16ToFloat32Workload<ConvertFp16ToFp32QueueDescriptor>(descriptor, info)
{
    this->m_Data.ValidateInputsOutputs("ClConvertFp16ToFp32Workload", 1, 1);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(this->m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(this->m_Data.m_Outputs[0])->GetTensor();

    // Create Proxy tensor and set the initial tensor handle to it
    m_InputProxy = std::make_unique<ICLTensorProxy>(&input);
    m_OutputProxy = std::make_unique<ICLTensorProxy>(&output);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClConvertFp16ToFp32Workload_configure");
        m_Layer.configure(clCompileContext, m_InputProxy.get(), m_OutputProxy.get(), g_AclConvertPolicy, 0);
    }
}

void ClConvertFp16ToFp32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClConvertFp16ToFp32Workload_Execute", this->GetGuid());
    RunClFunction(m_Layer, CHECK_LOCATION());
}

arm_compute::Status ClConvertFp16ToFp32WorkloadValidate(const TensorInfo& input, const TensorInfo& output)
{
    if (input.GetDataType() != DataType::Float16)
    {
        return arm_compute::Status(arm_compute::ErrorCode::RUNTIME_ERROR, "Input should be Float16");
    }
    if (output.GetDataType() != DataType::Float32)
    {
        return arm_compute::Status(arm_compute::ErrorCode::RUNTIME_ERROR, "Output should be Float32");
    }

    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    const arm_compute::Status aclStatus = arm_compute::CLDepthConvertLayer::validate(
        &aclInputInfo, &aclOutputInfo, g_AclConvertPolicy, 0);

    return aclStatus;
}

void ClConvertFp16ToFp32Workload::ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot)
{
    ITensorHandle* backupHandle = this->m_Data.m_Inputs[slot];
    this->m_Data.m_Inputs[slot] = tensorHandle;
    try
    {
        Reconfigure();
    }
    catch(armnn::UnimplementedException& e)
    {
        // Cannot reconfigure, revert the slot back and throw the exception.
        this->m_Data.m_Inputs[slot] = backupHandle;
        throw e;
    }
}

// Replace output tensor handle with the given TensorHandle
void ClConvertFp16ToFp32Workload::ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot)
{
    ITensorHandle* backupHandle = this->m_Data.m_Inputs[slot];
    this->m_Data.m_Inputs[slot] = tensorHandle;
    try
    {
        Reconfigure();
    }
    catch(armnn::UnimplementedException& e)
    {
        // Cannot reconfigure, revert the slot back and throw the exception.
        this->m_Data.m_Inputs[slot] = backupHandle;
        throw e;
    }
}

void ClConvertFp16ToFp32Workload::Reconfigure()
{
    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    m_InputProxy->set(&input);
    m_OutputProxy->set(&output);
}

} //namespace armnn
