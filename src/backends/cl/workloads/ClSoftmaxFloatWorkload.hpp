//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/CLFunctions.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>

#include <memory>

namespace armnn
{

class ClSoftmaxFloatWorkload : public FloatWorkload<SoftmaxQueueDescriptor>
{
public:
    ClSoftmaxFloatWorkload(const SoftmaxQueueDescriptor& descriptor, const WorkloadInfo& info,
                             std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);
    void Execute() const override;

private:
    mutable arm_compute::CLSoftmaxLayer m_SoftmaxLayer;
};

} //namespace armnn

