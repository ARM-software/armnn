//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "backendsCommon/Workload.hpp"
#include "backendsCommon/WorkloadData.hpp"

namespace armnn
{


class RefMeanFloat32Workload : public Float32Workload<MeanQueueDescriptor>
{
public:
    explicit RefMeanFloat32Workload (const MeanQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;
};

}//namespace armnn
