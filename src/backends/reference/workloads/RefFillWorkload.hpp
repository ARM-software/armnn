//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefFillWorkload : public BaseWorkload<FillQueueDescriptor>
{
public:
    using BaseWorkload<FillQueueDescriptor>::BaseWorkload;
    virtual void Execute() const override;
};

} //namespace armnn
