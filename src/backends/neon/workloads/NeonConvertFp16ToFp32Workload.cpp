//
// Copyright © 2017 Arm Ltd. All rights reserved.
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
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonConvertFp16ToFp32Workload_Execute");

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

} //namespace armnn
