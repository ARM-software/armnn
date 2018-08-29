//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"

namespace armnn
{

class RefDivisionUint8Workload : public Uint8Workload<DivisionQueueDescriptor>
{
public:
    using Uint8Workload<DivisionQueueDescriptor>::Uint8Workload;
    virtual void Execute() const override;
};

} //namespace armnn
