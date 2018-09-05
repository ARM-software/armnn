//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"

namespace armnn
{

class RefDivisionFloat32Workload : public Float32Workload<DivisionQueueDescriptor>
{
public:
    using Float32Workload<DivisionQueueDescriptor>::Float32Workload;
    virtual void Execute() const override;
};

} //namespace armnn
