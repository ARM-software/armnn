//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"

namespace armnn
{

class RefReshapeFloat32Workload : public Float32Workload<ReshapeQueueDescriptor>
{
public:
    using Float32Workload<ReshapeQueueDescriptor>::Float32Workload;
    virtual void Execute() const override;
};

} //namespace armnn
