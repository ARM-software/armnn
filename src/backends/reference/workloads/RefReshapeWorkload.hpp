//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefReshapeWorkload : public BaseWorkload<ReshapeQueueDescriptor>
{
public:
    using BaseWorkload<ReshapeQueueDescriptor>::BaseWorkload;
    virtual void Execute() const override;
};

} //namespace armnn
