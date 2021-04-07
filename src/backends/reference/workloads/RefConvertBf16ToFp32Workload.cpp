//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefConvertBf16ToFp32Workload.hpp"
#include "RefWorkloadUtils.hpp"

#include <armnnUtils/FloatingPointConverter.hpp>

#include <BFloat16.hpp>

namespace armnn
{

void RefConvertBf16ToFp32Workload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefConvertBf16ToFp32Workload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefConvertBf16ToFp32Workload::Execute(std::vector<ITensorHandle*> inputs,
                                           std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefConvertBf16ToFp32Workload_Execute");

    const BFloat16* const input = reinterpret_cast<const BFloat16*>(inputs[0]->Map());
    float* const output = reinterpret_cast<float*>(outputs[0]->Map());

    unsigned int numElements = GetTensorInfo(inputs[0]).GetNumElements();
    armnnUtils::FloatingPointConverter::ConvertBFloat16ToFloat32(input, numElements, output);
}

} //namespace armnn
