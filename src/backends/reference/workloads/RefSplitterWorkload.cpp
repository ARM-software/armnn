//
// Copyright © 2019-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefSplitterWorkload.hpp"
#include "Splitter.hpp"
#include "RefWorkloadUtils.hpp"
#include "Profiling.hpp"

namespace armnn
{

void RefSplitterWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefSplitterWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT_REF_NAME_GUID("RefSplitterWorkload_Execute");

    Split(m_Data, inputs, outputs);
}

} //namespace armnn
