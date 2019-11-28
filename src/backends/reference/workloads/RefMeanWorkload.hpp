//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

#include "Decoders.hpp"
#include "Encoders.hpp"

namespace armnn
{

class RefMeanWorkload : public BaseWorkload<MeanQueueDescriptor>
{
public:
    explicit RefMeanWorkload (const MeanQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;
};

} //namespace armnn
