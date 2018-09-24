//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"

namespace armnn
{

class RefFloorFloat32Workload : public Float32Workload<FloorQueueDescriptor>
{
public:
    using Float32Workload<FloorQueueDescriptor>::Float32Workload;
    virtual void Execute() const override;
};

} //namespace armnn
