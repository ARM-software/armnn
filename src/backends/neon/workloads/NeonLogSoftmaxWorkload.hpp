//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Descriptors.hpp>
#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>

#include <memory>

namespace armnn
{

arm_compute::Status NeonLogSoftmaxWorkloadValidate(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const LogSoftmaxDescriptor& descriptor);

class NeonLogSoftmaxWorkload : public NeonBaseWorkload<LogSoftmaxQueueDescriptor>
{
public:
    NeonLogSoftmaxWorkload(const LogSoftmaxQueueDescriptor& descriptor, const WorkloadInfo& info,
                           std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);
    virtual void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_LogSoftmaxLayer;
};

} //namespace armnn

