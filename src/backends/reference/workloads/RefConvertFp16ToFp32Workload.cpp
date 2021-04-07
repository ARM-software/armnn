//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefConvertFp16ToFp32Workload.hpp"
#include "RefWorkloadUtils.hpp"

#include <armnnUtils/FloatingPointConverter.hpp>

#include <Half.hpp>

namespace armnn
{

void RefConvertFp16ToFp32Workload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefConvertFp16ToFp32Workload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefConvertFp16ToFp32Workload::Execute(std::vector<ITensorHandle*> inputs,
                                           std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefConvertFp16ToFp32Workload_Execute");

    const Half* const input = reinterpret_cast<const Half*>(inputs[0]->Map());
    float* const output = reinterpret_cast<float*>(outputs[0]->Map());

    unsigned int numElements = GetTensorInfo(inputs[0]).GetNumElements();
    armnnUtils::FloatingPointConverter::ConvertFloat16To32(input, numElements, output);
}

} //namespace armnn
