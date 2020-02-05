//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLActivationLayer.h>

namespace armnn
{
arm_compute::Status ClActivationWorkloadValidate(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const ActivationDescriptor& descriptor);

class ClActivationWorkload : public BaseWorkload<ActivationQueueDescriptor>
{
public:
    ClActivationWorkload(const ActivationQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::CLActivationLayer m_ActivationLayer;
};

} //namespace armnn
