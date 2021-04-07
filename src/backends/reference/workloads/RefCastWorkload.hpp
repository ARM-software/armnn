//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include "RefWorkloadUtils.hpp"

namespace armnn
{


class RefCastWorkload : public BaseWorkload<CastQueueDescriptor>
{
public:
    using BaseWorkload<CastQueueDescriptor>::BaseWorkload;
    void Execute() const override;
};

} //namespace armnn

