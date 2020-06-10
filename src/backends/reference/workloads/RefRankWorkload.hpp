//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

#include "RefWorkloadUtils.hpp"

namespace armnn
{

struct RefRankWorkload : public BaseWorkload<RankQueueDescriptor>
{
public:
    using BaseWorkload<RankQueueDescriptor>::BaseWorkload;
    virtual void Execute() const override
    {
        const int32_t rank = static_cast<int32_t>(GetTensorInfo(m_Data.m_Inputs[0]).GetNumDimensions());

        std::memcpy(GetOutputTensorData<void>(0, m_Data), &rank, sizeof(int32_t));
    }
};

} //namespace armnn




