//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/IFunction.h>

namespace armnn
{

arm_compute::Status NeonFusedWorkloadValidate(const std::vector<std::reference_wrapper<TensorInfo>>& inputInfos,
                                              const std::vector<std::reference_wrapper<TensorInfo>>& outputInfos,
                                              const FusedDescriptor& fusedDescriptor,
                                              const ActivationDescriptor* activationDescriptor = nullptr);

class NeonFusedWorkload : public NeonBaseWorkload<FusedQueueDescriptor>
{
public:
    NeonFusedWorkload(const FusedQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_FusedLayer;
};

} //namespace armnn



