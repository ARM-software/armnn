//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonConvertFp32ToBf16Workload.hpp"

#include <BFloat16.hpp>
#include <Profiling.hpp>

#include <armnnUtils/FloatingPointConverter.hpp>

#include <backendsCommon/WorkloadUtils.hpp>

namespace armnn
{

NeonConvertFp32ToBf16Workload::NeonConvertFp32ToBf16Workload(const ConvertFp32ToBf16QueueDescriptor& descriptor,
                                                             const WorkloadInfo& info)
    : Float32ToBFloat16Workload<ConvertFp32ToBf16QueueDescriptor>(descriptor, info)
{
    this->m_Data.ValidateInputsOutputs("NeonConvertFp32ToBf16Workload", 1, 1);
    GatherTensorHandlePairs(descriptor, m_TensorHandlePairs);
}

void NeonConvertFp32ToBf16Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonConvertFp32ToBf16Workload_Execute");

    auto convertFunc = [](uint8_t* dst, const uint8_t* src, size_t size)
        {
            auto input = reinterpret_cast<const float*>(src);
            auto output = reinterpret_cast<BFloat16*>(dst);
            size_t numElements = size/2; // 2 bytes per bf16
            armnnUtils::FloatingPointConverter::ConvertFloat32ToBFloat16(input, numElements, output);
        };

    for (const auto& pair : m_TensorHandlePairs)
    {
        CopyTensorContentsGeneric(pair.first, pair.second, convertFunc);
    }
}

} //namespace armnn
