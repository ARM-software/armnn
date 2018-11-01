//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefMergerFloat32Workload : public Float32Workload<MergerQueueDescriptor>
{
public:
    using Float32Workload<MergerQueueDescriptor>::Float32Workload;
    virtual void Execute() const override;
};

} //namespace armnn
