//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class SampleDynamicAdditionWorkload : public BaseWorkload<AdditionQueueDescriptor>
{
public:
    SampleDynamicAdditionWorkload(const AdditionQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;
};

} // namespace armnn
