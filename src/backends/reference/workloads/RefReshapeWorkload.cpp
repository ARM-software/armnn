//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefReshapeWorkload.hpp"
#include "RefWorkloadUtils.hpp"
#include "Profiling.hpp"

#include <cstring>

namespace armnn
{

void RefReshapeWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefReshapeWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefReshapeWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefReshapeWorkload_Execute");

    void* output =  outputs[0]->Map();
    const void* input =  inputs[0]->Map();
    unsigned int numBytes = GetTensorInfo(inputs[0]).GetNumBytes();
    memcpy(output, input, numBytes);
}

} //namespace armnn
