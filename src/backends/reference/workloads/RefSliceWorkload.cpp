//
// Copyright © 2019,2020-2024 Arm Ltd and Contributors. All rights reserved.
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

void RefSliceWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT_REF_NAME_GUID("RefSliceWorkload_Execute");

    const TensorInfo& inputInfo  = GetTensorInfo(inputs[0]);

    Slice(inputInfo,
          m_Data.m_Parameters,
          inputs[0]->Map(),
          outputs[0]->Map(),
          GetDataTypeSize(inputInfo.GetDataType()));
}

} // namespace armnn
