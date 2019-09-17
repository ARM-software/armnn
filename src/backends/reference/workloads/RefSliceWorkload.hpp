//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefSliceWorkload : public BaseWorkload<SliceQueueDescriptor>
{
public:
    using BaseWorkload<SliceQueueDescriptor>::BaseWorkload;

    virtual void Execute() const override;
};

} // namespace armnn
