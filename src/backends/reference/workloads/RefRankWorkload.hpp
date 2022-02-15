//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

#include "RefWorkloadUtils.hpp"

namespace armnn
{

struct RefRankWorkload : public RefBaseWorkload<RankQueueDescriptor>
{
public:
    using RefBaseWorkload<RankQueueDescriptor>::RefBaseWorkload;
    virtual void Execute() const override
    {
        Execute(m_Data.m_Inputs, m_Data.m_Outputs);

    }
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)  override
    {
        Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
    }

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
    {
        const int32_t rank = static_cast<int32_t>(GetTensorInfo(inputs[0]).GetNumDimensions());

        std::memcpy(outputs[0]->Map(), &rank, sizeof(int32_t));
        outputs[0]->Unmap();
    }
};

} //namespace armnn




