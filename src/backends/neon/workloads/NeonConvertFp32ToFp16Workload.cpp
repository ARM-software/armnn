//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonConvertFp32ToFp16Workload.hpp"

#include <Half.hpp>
#include <Profiling.hpp>

#include <armnnUtils/FloatingPointConverter.hpp>

#include <backendsCommon/WorkloadUtils.hpp>

namespace armnn
{

NeonConvertFp32ToFp16Workload::NeonConvertFp32ToFp16Workload(const ConvertFp32ToFp16QueueDescriptor& descriptor,
                                                             const WorkloadInfo& info)
    : Float32ToFloat16Workload<ConvertFp32ToFp16QueueDescriptor>(descriptor, info)
{
    this->m_Data.ValidateInputsOutputs("NeonConvertFp32ToFp16Workload", 1, 1);
    GatherTensorHandlePairs(descriptor, m_TensorHandlePairs);
}

void NeonConvertFp32ToFp16Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonConvertFp32ToFp16Workload_Execute");

    auto convertFunc = [](uint8_t* dst, const uint8_t* src, size_t size)
        {
            auto input = reinterpret_cast<const float*>(src);
            auto output = reinterpret_cast<Half*>(dst);
            size_t numElements = size/2; // 2 bytes per fp16
            armnnUtils::FloatingPointConverter::ConvertFloat32To16(input, numElements, output);
        };

    for (const auto& pair : m_TensorHandlePairs)
    {
        CopyTensorContentsGeneric(pair.first, pair.second, convertFunc);
    }
}

} //namespace armnn
