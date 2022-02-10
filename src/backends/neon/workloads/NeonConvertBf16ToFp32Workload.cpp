//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonConvertBf16ToFp32Workload.hpp"

#include <armnnUtils/FloatingPointConverter.hpp>

#include <BFloat16.hpp>

#include <backendsCommon/WorkloadUtils.hpp>

namespace armnn
{

NeonConvertBf16ToFp32Workload::NeonConvertBf16ToFp32Workload(const ConvertBf16ToFp32QueueDescriptor& descriptor,
                                                             const WorkloadInfo& info)
     : BFloat16ToFloat32Workload<ConvertBf16ToFp32QueueDescriptor>(descriptor, info)
{
    this->m_Data.ValidateInputsOutputs("NeonConvertBf16ToFp32Workload", 1, 1);
    GatherTensorHandlePairs(descriptor, m_TensorHandlePairs);
}

void NeonConvertBf16ToFp32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonConvertBf16ToFp32Workload_Execute", this->GetGuid());

    auto convertFunc = [](uint8_t* dst, const uint8_t* src, size_t size)
        {
            auto input = reinterpret_cast<const BFloat16*>(src);
            auto output = reinterpret_cast<float*>(dst);
            size_t numElements = size/2; // 2 bytes per Bf16
            armnnUtils::FloatingPointConverter::ConvertBFloat16ToFloat32(input, numElements, output);
        };

    for (const auto& pair : m_TensorHandlePairs)
    {
        CopyTensorContentsGeneric(pair.first, pair.second, convertFunc);
    }
}

void NeonConvertBf16ToFp32Workload::ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot)
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
void NeonConvertBf16ToFp32Workload::ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot)
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

void NeonConvertBf16ToFp32Workload::Reconfigure()
{
    throw armnn::UnimplementedException("Reconfigure not implemented for this workload");
}

} //namespace armnn
