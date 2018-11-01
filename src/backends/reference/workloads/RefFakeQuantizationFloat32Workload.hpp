//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefFakeQuantizationFloat32Workload : public Float32Workload<FakeQuantizationQueueDescriptor>
{
public:
    using Float32Workload<FakeQuantizationQueueDescriptor>::Float32Workload;
    virtual void Execute() const override;
};

} //namespace armnn
