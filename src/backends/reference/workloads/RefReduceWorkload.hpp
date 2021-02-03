//
// Copyright © 2020 Samsung Electronics Co Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefReduceWorkload : public BaseWorkload<ReduceQueueDescriptor>
{
public:
    explicit RefReduceWorkload(const ReduceQueueDescriptor& descriptor,
                               const WorkloadInfo& info);

    virtual void Execute() const override;
};

} //namespace armnn
