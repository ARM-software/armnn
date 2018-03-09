//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/Workload.hpp>

namespace armnn
{
// Base class template providing an implementation of the Merger layer common to all data types
template <armnn::DataType DataType>
class NeonBaseMergerWorkload : public TypedWorkload<MergerQueueDescriptor, DataType>
{
public:
    using TypedWorkload<MergerQueueDescriptor, DataType>::TypedWorkload;

    virtual void Execute() const override
    {
        // With subtensors, merger is a no-op
    }
};

} //namespace armnn
