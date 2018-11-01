//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefPooling2dFloat32Workload : public Float32Workload<Pooling2dQueueDescriptor>
{
public:
    using Float32Workload<Pooling2dQueueDescriptor>::Float32Workload;
    virtual void Execute() const override;
};

} //namespace armnn
