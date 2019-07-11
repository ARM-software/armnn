//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefStackWorkload : public BaseWorkload<StackQueueDescriptor>
{
public:
    explicit RefStackWorkload(const StackQueueDescriptor& descriptor,
                              const WorkloadInfo& info);
    virtual void Execute() const override;
};

} // namespace armnn
