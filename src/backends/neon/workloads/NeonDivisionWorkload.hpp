//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <neon/workloads/NeonWorkloadUtils.hpp>

#include <arm_compute/runtime/NEON/functions/NEElementwiseOperations.h>

namespace armnn
{

arm_compute::Status NeonDivisionWorkloadValidate(const TensorInfo& input0,
                                                 const TensorInfo& input1,
                                                 const TensorInfo& output,
                                                 const ActivationDescriptor* activationDescriptor = nullptr);

class NeonDivisionWorkload : public NeonBaseWorkload<DivisionQueueDescriptor>
{
public:
    NeonDivisionWorkload(const DivisionQueueDescriptor& descriptor, const WorkloadInfo& info);

    virtual void Execute() const override;

private:
    mutable arm_compute::NEElementwiseDivision m_DivLayer;
};

} //namespace armnn
