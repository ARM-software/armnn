//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefPadWorkload.hpp"

#include "MirrorPad.hpp"
#include "Pad.hpp"
#include "Profiling.hpp"
#include "RefWorkloadUtils.hpp"

namespace armnn
{

void RefPadWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefPadWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefPadWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefPadWorkload_Execute");

    const TensorInfo& inputInfo  = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    PaddingMode paddingMode = m_Data.m_Parameters.m_PaddingMode;
    if (paddingMode == PaddingMode::Constant)
    {
        armnn::Pad(inputInfo, outputInfo, inputs[0], outputs[0], m_Data);
    }
    else if(paddingMode == PaddingMode::Reflect || paddingMode == PaddingMode::Symmetric)
    {
        armnn::MirrorPad(inputInfo, outputInfo, inputs[0], outputs[0], m_Data);
    }
    else
    {
        throw InvalidArgumentException("Padding mode not supported.");
    }
}

} //namespace armnn