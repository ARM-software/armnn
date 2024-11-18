//
// Copyright © 2017,2019-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefConcatWorkload.hpp"
#include "Concatenate.hpp"
#include "Profiling.hpp"
#include "RefWorkloadUtils.hpp"

namespace armnn
{

void RefConcatWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefConcatWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT_REF_NAME_GUID("RefConcatWorkload_Execute");
    Concatenate(m_Data, inputs, outputs);
}

} //namespace armnn
