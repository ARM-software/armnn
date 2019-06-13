//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefPreluWorkload : public BaseWorkload<PreluQueueDescriptor>
{
public:
    explicit RefPreluWorkload(const PreluQueueDescriptor& descriptor,
                              const WorkloadInfo& info);
    virtual void Execute() const override;
};

} // namespace armnn
