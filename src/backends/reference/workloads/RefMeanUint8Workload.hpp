//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "backendsCommon/Workload.hpp"
#include "backendsCommon/WorkloadData.hpp"

namespace armnn
{

class RefMeanUint8Workload : public Uint8Workload<MeanQueueDescriptor>
{
public:
    explicit RefMeanUint8Workload (const MeanQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;
};

} //namespace armnn
