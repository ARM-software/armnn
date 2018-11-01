//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

namespace armnn
{

class NeonSplitterWorkload : public BaseWorkload<SplitterQueueDescriptor>
{
public:
    using BaseWorkload<SplitterQueueDescriptor>::BaseWorkload;

    virtual void Execute() const override
    {
        // With subtensors, splitter is a no-op.
    }
};

} //namespace armnn
