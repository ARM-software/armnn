//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"

namespace armnn
{

class RefSubtractionUint8Workload : public Uint8Workload<SubtractionQueueDescriptor>
{
public:
    using Uint8Workload<SubtractionQueueDescriptor>::Uint8Workload;
    virtual void Execute() const override;
};

} //namespace armnn
