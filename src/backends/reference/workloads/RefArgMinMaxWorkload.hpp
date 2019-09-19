//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{
class RefArgMinMaxWorkload : public BaseWorkload<ArgMinMaxQueueDescriptor>
{
public:
    explicit RefArgMinMaxWorkload(const ArgMinMaxQueueDescriptor& descriptor,
                                  const WorkloadInfo& info);

    virtual void Execute() const override;
};
} //namespace armnn