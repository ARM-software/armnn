//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/neon/workloads/NeonWorkloadUtils.hpp>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>

namespace armnn
{

class NeonSoftmaxUint8Workload : public Uint8Workload<SoftmaxQueueDescriptor>
{
public:
    NeonSoftmaxUint8Workload(const SoftmaxQueueDescriptor& descriptor, const WorkloadInfo& info,
                             std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);
    virtual void Execute() const override;

private:
    mutable arm_compute::NESoftmaxLayer m_SoftmaxLayer;
};

} //namespace armnn

