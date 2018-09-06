//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"

namespace armnn
{

class RefSubtractionFloat32Workload : public Float32Workload<SubtractionQueueDescriptor>
{
public:
    using Float32Workload<SubtractionQueueDescriptor>::Float32Workload;
    virtual void Execute() const override;
};

} //namespace armnn
