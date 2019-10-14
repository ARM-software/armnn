//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefLogSoftmaxWorkload : public BaseWorkload<LogSoftmaxQueueDescriptor>
{
public:
    using BaseWorkload<LogSoftmaxQueueDescriptor>::BaseWorkload;
    virtual void Execute() const override;
};

} // namespace armnn
