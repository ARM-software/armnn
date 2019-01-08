//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>

#include <memory>

namespace armnn
{

class NeonSoftmaxUint8Workload : public Uint8Workload<SoftmaxQueueDescriptor>
{
public:
    NeonSoftmaxUint8Workload(const SoftmaxQueueDescriptor& descriptor, const WorkloadInfo& info,
                             std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);
    virtual void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_SoftmaxLayer;
};

} //namespace armnn

