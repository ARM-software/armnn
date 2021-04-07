//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefDepthToSpaceWorkload.hpp"

#include "DepthToSpace.hpp"
#include "RefWorkloadUtils.hpp"

namespace armnn
{

void RefDepthToSpaceWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefDepthToSpaceWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefDepthToSpaceWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDepthToSpaceWorkload_Execute");

    const TensorInfo inputInfo = GetTensorInfo(inputs[0]);

    DepthToSpace(inputInfo,
                 m_Data.m_Parameters,
                 inputs[0]->Map(),
                 outputs[0]->Map(),
                 GetDataTypeSize(inputInfo.GetDataType()));
}

} // namespace armnn
