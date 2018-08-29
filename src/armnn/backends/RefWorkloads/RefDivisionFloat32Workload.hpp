//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
