//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefBatchToSpaceNdUint8Workload : public Uint8Workload<BatchToSpaceNdQueueDescriptor>
{

public:
    using Uint8Workload<BatchToSpaceNdQueueDescriptor>::Uint8Workload;

    virtual void Execute() const override;
};

} // namespace armnn