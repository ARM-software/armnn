//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Descriptors.hpp>

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>
#include <arm_compute/runtime/CL/functions/CLSoftmaxLayer.h>

#include "ClBaseWorkload.hpp"

namespace armnn
{

arm_compute::Status ClSoftmaxWorkloadValidate(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const SoftmaxDescriptor& descriptor);

class ClSoftmaxWorkload : public ClBaseWorkload<SoftmaxQueueDescriptor>
{
public:
    ClSoftmaxWorkload(const SoftmaxQueueDescriptor& descriptor,
                      const WorkloadInfo& info,
                      std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager,
                      const arm_compute::CLCompileContext& clCompileContext);
    void Execute() const override;

private:
    mutable arm_compute::CLSoftmaxLayer m_SoftmaxLayer;
};

} // namespace armnn
