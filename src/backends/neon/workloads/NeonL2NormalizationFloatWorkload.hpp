//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>

#include <memory>

namespace armnn
{

arm_compute::Status NeonL2NormalizationWorkloadValidate(const TensorInfo& input,
                                                        const TensorInfo& output,
                                                        const L2NormalizationDescriptor& descriptor);

class NeonL2NormalizationFloatWorkload : public FloatWorkload<L2NormalizationQueueDescriptor>
{
public:
    NeonL2NormalizationFloatWorkload(const L2NormalizationQueueDescriptor& descriptor, const WorkloadInfo& info,
                                     std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);
    virtual void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_Layer;
};

} //namespace armnn

