//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefSliceWorkload.hpp"

#include "RefWorkloadUtils.hpp"
#include "Slice.hpp"

#include <Profiling.hpp>

namespace armnn
{

void RefSliceWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefSliceWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefSliceWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefSliceWorkload_Execute");

    const TensorInfo& inputInfo  = GetTensorInfo(inputs[0]);

    Slice(inputInfo,
          m_Data.m_Parameters,
          inputs[0]->Map(),
          outputs[0]->Map(),
          GetDataTypeSize(inputInfo.GetDataType()));
}

} // namespace armnn
