//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonConvertFp16ToFp32Workload.hpp"

#include <armnnUtils/FloatingPointConverter.hpp>

#include <Half.hpp>

#include <backendsCommon/WorkloadUtils.hpp>

namespace armnn
{

NeonConvertFp16ToFp32Workload::NeonConvertFp16ToFp32Workload(const ConvertFp16ToFp32QueueDescriptor& descriptor,
                                                             const WorkloadInfo& info)
     : Float16ToFloat32Workload<ConvertFp16ToFp32QueueDescriptor>(descriptor, info)
{
    this->m_Data.ValidateInputsOutputs("NeonConvertFp16ToFp32Workload", 1, 1);
    GatherTensorHandlePairs(descriptor, m_TensorHandlePairs);
}

void NeonConvertFp16ToFp32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonConvertFp16ToFp32Workload_Execute", this->GetGuid());

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
