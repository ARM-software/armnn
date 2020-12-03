//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

#include "NeonWorkloadUtils.hpp"

namespace armnn
{

struct NeonRankWorkload : public BaseWorkload<RankQueueDescriptor>
{
public:
    using BaseWorkload<RankQueueDescriptor>::BaseWorkload;
    virtual void Execute() const override
    {
        const NeonTensorHandle* neonTensorHandle = PolymorphicDowncast<const NeonTensorHandle*>(m_Data.m_Inputs[0]);
        const int32_t rank = static_cast<int32_t>(neonTensorHandle->GetShape().GetNumDimensions());

        std::memcpy(GetOutputTensorData<void>(0, m_Data), &rank, sizeof(int32_t));
        m_Data.m_Outputs[0]->Unmap();
    }
};

} //namespace armnn
