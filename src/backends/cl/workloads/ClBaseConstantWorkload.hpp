//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/Workload.hpp>

#include <arm_compute/runtime/CL/CLFunctions.h>

namespace armnn
{
template <armnn::DataType... DataTypes>
class ClBaseConstantWorkload : public TypedWorkload<ConstantQueueDescriptor, DataTypes...>
{
public:
    ClBaseConstantWorkload(const ConstantQueueDescriptor& descriptor, const WorkloadInfo& info)
        : TypedWorkload<ConstantQueueDescriptor, DataTypes...>(descriptor, info)
        , m_RanOnce(false)
    {
    }

    void Execute() const override;

private:
    mutable bool m_RanOnce;
};

} //namespace armnn
