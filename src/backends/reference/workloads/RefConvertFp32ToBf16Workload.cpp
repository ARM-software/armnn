//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefConvertFp32ToBf16Workload.hpp"
#include "RefWorkloadUtils.hpp"

#include <armnnUtils/FloatingPointConverter.hpp>

#include <BFloat16.hpp>

namespace armnn
{

void RefConvertFp32ToBf16Workload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefConvertFp32ToBf16Workload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefConvertFp32ToBf16Workload::Execute(std::vector<ITensorHandle*> inputs,
                                           std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefConvertFp32ToBf16Workload_Execute");

    const float* const input = reinterpret_cast<const float*>(inputs[0]->Map());
    BFloat16* const output = reinterpret_cast<BFloat16*>(outputs[0]->Map());

    unsigned int numElements = GetTensorInfo(inputs[0]).GetNumElements();
    armnnUtils::FloatingPointConverter::ConvertFloat32ToBFloat16(input, numElements, output);
}

} //namespace armnn
