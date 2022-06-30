//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefConvertFp32ToFp16Workload.hpp"
#include "RefWorkloadUtils.hpp"
#include "Profiling.hpp"

#include <armnnUtils/FloatingPointConverter.hpp>

#include <Half.hpp>

namespace armnn
{

void RefConvertFp32ToFp16Workload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefConvertFp32ToFp16Workload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefConvertFp32ToFp16Workload::Execute(std::vector<ITensorHandle*> inputs,
                                           std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefConvertFp32ToFp16Workload_Execute");

    const float* const input = reinterpret_cast<const float*>(inputs[0]->Map());
    Half*  const output = reinterpret_cast<Half*>(outputs[0]->Map());

    // convert Fp32 input to Fp16 output
    unsigned int numElements = GetTensorInfo(inputs[0]).GetNumElements();
    armnnUtils::FloatingPointConverter::ConvertFloat32To16(input, numElements, output);
}

} //namespace armnn
