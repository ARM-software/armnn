//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/Workload.hpp>
#include <backends/neon/workloads/NeonWorkloadUtils.hpp>

namespace armnn
{

// Base class template providing an implementation of the Splitter layer common to all data types.
template <armnn::DataType... DataTypes>
class NeonBaseSplitterWorkload : public TypedWorkload<SplitterQueueDescriptor, DataTypes...>
{
public:
    using TypedWorkload<SplitterQueueDescriptor, DataTypes...>::TypedWorkload;

    virtual void Execute() const override
    {
        // With subtensors, splitter is a no-op.
    }
};

} //namespace armnn
