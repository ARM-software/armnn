//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/Workload.hpp>

namespace armnn
{

// Base class template providing an implementation of the Splitter layer common to all data types.
class ClSplitterWorkload : public BaseWorkload<SplitterQueueDescriptor>
{
public:
    using BaseWorkload<SplitterQueueDescriptor>::BaseWorkload;

    void Execute() const override
    {
        // With subtensors, splitter is a no-op.
    }
};

} //namespace armnn
