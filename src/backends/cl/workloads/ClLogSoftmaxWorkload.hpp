//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Descriptors.hpp>

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>
#include <arm_compute/runtime/CL/functions/CLSoftmaxLayer.h>


#include <backendsCommon/Workload.hpp>

namespace armnn
{

arm_compute::Status ClLogSoftmaxWorkloadValidate(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const LogSoftmaxDescriptor& descriptor);

class ClLogSoftmaxWorkload : public BaseWorkload<LogSoftmaxQueueDescriptor>
{
public:
    ClLogSoftmaxWorkload(const LogSoftmaxQueueDescriptor& descriptor, const WorkloadInfo& info,
                         std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);
    void Execute() const override;

private:
    mutable arm_compute::CLLogSoftmaxLayer m_LogSoftmaxLayer;
};

} // namespace armnn
