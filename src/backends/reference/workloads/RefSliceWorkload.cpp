//
// Copyright © 2019 Arm Ltd. All rights reserved.
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
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefSliceWorkload_Execute");

    const TensorInfo& inputInfo  = GetTensorInfo(m_Data.m_Inputs[0]);

    Slice(inputInfo,
          m_Data.m_Parameters,
          m_Data.m_Inputs[0]->Map(),
          m_Data.m_Outputs[0]->Map(),
          GetDataTypeSize(inputInfo.GetDataType()));
}

} // namespace armnn
