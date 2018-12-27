//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefRsqrtFloat32Workload : public Float32Workload<RsqrtQueueDescriptor>
{
public:
    using Float32Workload<RsqrtQueueDescriptor>::Float32Workload;
    virtual void Execute() const override;
};

} //namespace armnn
