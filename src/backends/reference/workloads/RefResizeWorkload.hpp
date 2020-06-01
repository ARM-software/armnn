//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefResizeWorkload : public BaseWorkload<ResizeQueueDescriptor>
{
public:
    using BaseWorkload<ResizeQueueDescriptor>::BaseWorkload;
    virtual void Execute() const override;
};

} //namespace armnn
