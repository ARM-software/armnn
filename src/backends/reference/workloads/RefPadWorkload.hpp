//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"

namespace armnn
{

class RefPadWorkload : public BaseWorkload<PadQueueDescriptor>
{
public:
    explicit RefPadWorkload (const PadQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;
};

} //namespace armnn
