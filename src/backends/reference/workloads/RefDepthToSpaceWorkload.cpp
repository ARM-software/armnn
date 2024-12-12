//
// Copyright © 2019,2021-2024 Arm Ltd and Contributors. All rights reserved.
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

void RefDepthToSpaceWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT_REF_NAME_GUID("RefDepthToSpaceWorkload_Execute");
    const TensorInfo inputInfo = GetTensorInfo(inputs[0]);

    DepthToSpace(inputInfo,
                 m_Data.m_Parameters,
                 inputs[0]->Map(),
                 outputs[0]->Map(),
                 GetDataTypeSize(inputInfo.GetDataType()));
}

} // namespace armnn
