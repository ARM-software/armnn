//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefInstanceNormalizationWorkload : public BaseWorkload<InstanceNormalizationQueueDescriptor>
{
public:
    explicit RefInstanceNormalizationWorkload(const InstanceNormalizationQueueDescriptor& descriptor,
                                              const WorkloadInfo& info);
    virtual void Execute() const override;
};

} //namespace armnn
