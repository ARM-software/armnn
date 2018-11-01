//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/CLFunctions.h>

namespace armnn
{
class ClConstantWorkload : public BaseWorkload<ConstantQueueDescriptor>
{
public:
    ClConstantWorkload(const ConstantQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable bool m_RanOnce;
};

} //namespace armnn
