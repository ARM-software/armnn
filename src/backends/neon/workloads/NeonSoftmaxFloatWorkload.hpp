//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <neon/workloads/NeonWorkloadUtils.hpp>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>

#include <memory>

namespace armnn
{

class NeonSoftmaxFloatWorkload : public FloatWorkload<SoftmaxQueueDescriptor>
{
public:
    NeonSoftmaxFloatWorkload(const SoftmaxQueueDescriptor& descriptor, const WorkloadInfo& info,
                             std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);
    virtual void Execute() const override;

private:
    mutable arm_compute::NESoftmaxLayer m_SoftmaxLayer;
};

} //namespace armnn

