//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

#include "arm_compute/runtime/MemoryManagerOnDemand.h"

#include <memory>

namespace armnn
{

class ClSoftmaxFloat32Workload : public FloatWorkload<SoftmaxQueueDescriptor>
{
public:
    ClSoftmaxFloat32Workload(const SoftmaxQueueDescriptor& descriptor, const WorkloadInfo& info,
                             std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);
    void Execute() const override;

private:
    mutable arm_compute::CLSoftmaxLayer m_SoftmaxLayer;
};

} //namespace armnn

