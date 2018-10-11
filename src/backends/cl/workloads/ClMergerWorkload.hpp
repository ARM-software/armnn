//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/Workload.hpp>

namespace armnn
{

class ClMergerWorkload : public BaseWorkload<MergerQueueDescriptor>
{
public:
    using BaseWorkload<MergerQueueDescriptor>::BaseWorkload;

    void Execute() const override
    {
        // With subtensors, merger is a no-op.
    }
};

} //namespace armnn
