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

class ClFullyConnectedFloat32Workload : public armnn::Float32Workload<armnn::FullyConnectedQueueDescriptor>
{
public:
    ClFullyConnectedFloat32Workload(const armnn::FullyConnectedQueueDescriptor& descriptor,
                                    const armnn::WorkloadInfo& info,
                                    std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);

    using armnn::Float32Workload<armnn::FullyConnectedQueueDescriptor>::m_Data;
    void Execute() const override;

private:
    mutable arm_compute::CLFullyConnectedLayer m_FullyConnected;
    arm_compute::CLTensor                      m_WeightsTensor;
    arm_compute::CLTensor                      m_BiasesTensor;
};

} //namespace armnn
