//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <neon/workloads/NeonWorkloadUtils.hpp>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>

namespace armnn
{

arm_compute::Status NeonNormalizationWorkloadValidate(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const NormalizationDescriptor& descriptor);

class NeonNormalizationFloatWorkload : public FloatWorkload<NormalizationQueueDescriptor>
{
public:
    NeonNormalizationFloatWorkload(const NormalizationQueueDescriptor& descriptor, const WorkloadInfo& info,
                                   std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);
    virtual void Execute() const override;

private:
    mutable arm_compute::NENormalizationLayer m_NormalizationLayer;
};

} //namespace armnn




