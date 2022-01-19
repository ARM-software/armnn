//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{

struct ClRankWorkload : public ClBaseWorkload<RankQueueDescriptor>
{
public:
    using ClBaseWorkload<RankQueueDescriptor>::ClBaseWorkload;
    virtual void Execute() const override
    {
        const ClTensorHandle* clTensorHandle = PolymorphicDowncast<const ClTensorHandle*>(m_Data.m_Inputs[0]);
        const int32_t rank = static_cast<int32_t>(clTensorHandle->GetShape().GetNumDimensions());

        std::memcpy(GetOutputTensorData<void>(0, m_Data), &rank, sizeof(int32_t));
        m_Data.m_Outputs[0]->Unmap();
    }
};

} //namespace armnn
