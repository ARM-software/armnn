//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/neon/workloads/NeonWorkloadUtils.hpp>
#include <backends/Workload.hpp>

namespace armnn
{
// Base class template providing an implementation of the Merger layer common to all data types.
template <armnn::DataType... DataTypes>
class NeonBaseMergerWorkload : public TypedWorkload<MergerQueueDescriptor, DataTypes...>
{
public:
    using TypedWorkload<MergerQueueDescriptor, DataTypes...>::TypedWorkload;

    virtual void Execute() const override
    {
        // With subtensors, merger is a no-op.
    }
};

} //namespace armnn
