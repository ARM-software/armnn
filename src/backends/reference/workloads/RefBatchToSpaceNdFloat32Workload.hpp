//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn {

class RefBatchToSpaceNdFloat32Workload : public Float32Workload<BatchToSpaceNdQueueDescriptor>
{

public:
    using Float32Workload<BatchToSpaceNdQueueDescriptor>::Float32Workload;

    virtual void Execute() const override;
};

} // namespace armnn