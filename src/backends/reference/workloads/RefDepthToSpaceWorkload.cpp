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
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDepthToSpaceWorkload_Execute");

    const TensorInfo inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);

    DepthToSpace(inputInfo,
                 m_Data.m_Parameters,
                 m_Data.m_Inputs[0]->Map(),
                 m_Data.m_Outputs[0]->Map(),
                 GetDataTypeSize(inputInfo.GetDataType()));
}

} // namespace armnn
