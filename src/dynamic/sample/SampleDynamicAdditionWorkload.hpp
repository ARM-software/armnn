//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadData.hpp>

namespace sdb // sample dynamic backend
{

class SampleDynamicAdditionWorkload : public armnn::BaseWorkload<armnn::AdditionQueueDescriptor>
{
public:
    SampleDynamicAdditionWorkload(const armnn::AdditionQueueDescriptor& descriptor, const armnn::WorkloadInfo& info);

    void Execute() const override;
};

} // namespace sdb
