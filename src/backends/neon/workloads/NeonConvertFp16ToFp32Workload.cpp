//
// Copyright © 2017-2019,2021-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonConvertFp16ToFp32Workload.hpp"

#include <armnnUtils/FloatingPointConverter.hpp>

#include <Half.hpp>

#include <backendsCommon/WorkloadUtils.hpp>

static constexpr arm_compute::ConvertPolicy g_AclConvertPolicy = arm_compute::ConvertPolicy::SATURATE;

namespace armnn
{

arm_compute::Status NeonConvertFp16ToFp32WorkloadValidate(const TensorInfo& input, const TensorInfo& output)
{
    // Fallback to portable software implementation if Compute Library NECast won't work, so
    // this method always returns success

    armnn::IgnoreUnused(input);
    armnn::IgnoreUnused(output);
    return arm_compute::Status();
}

NeonConvertFp16ToFp32Workload::NeonConvertFp16ToFp32Workload(const ConvertFp16ToFp32QueueDescriptor& descriptor,
                                                             const WorkloadInfo& info)
     : Float16ToFloat32Workload<ConvertFp16ToFp32QueueDescriptor>(descriptor, info)
{
    this->m_Data.ValidateInputsOutputs("NeonConvertFp16ToFp32Workload", 1, 1);

    arm_compute::ITensor& input  = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    if (arm_compute::NECast::validate(input.info(), output.info(), g_AclConvertPolicy))
    {
        // Use NECast if supported (needs hardware support for FP16)
        m_Cast.reset(new arm_compute::NECast());
        m_Cast->configure(&input, &output, g_AclConvertPolicy);
    }
    else
    {
        // Else use software implementation using Half.hpp
        GatherTensorHandlePairs(descriptor, m_TensorHandlePairs);
    }
}

void NeonConvertFp16ToFp32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_NAME_GUID("NeonConvertFp16ToFp32Workload_Execute");

    if (m_Cast)
    {
        // Use NECast if supported and initialised
        m_Cast->run();
    }
    else
    {
        // Else use softare implementation using Half.hpp
        auto convertFunc = [](uint8_t* dst, const uint8_t* src, size_t size)
        {
            auto input = reinterpret_cast<const Half*>(src);
            auto output = reinterpret_cast<float*>(dst);
            size_t numElements = size/2; // 2 bytes per fp16
            armnnUtils::FloatingPointConverter::ConvertFloat16To32(input, numElements, output);
        };

        for (const auto& pair : m_TensorHandlePairs)
        {
            CopyTensorContentsGeneric(pair.first, pair.second, convertFunc);
        }
    }
}

void NeonConvertFp16ToFp32Workload::ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot)
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
void NeonConvertFp16ToFp32Workload::ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot)
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

void NeonConvertFp16ToFp32Workload::Reconfigure()
{
    throw armnn::UnimplementedException("Reconfigure not implemented for this workload");
}

} //namespace armnn
