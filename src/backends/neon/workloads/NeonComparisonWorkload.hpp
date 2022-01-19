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

arm_compute::Status NeonComparisonWorkloadValidate(const TensorInfo& input0,
                                                   const TensorInfo& input1,
                                                   const TensorInfo& output,
                                                   const ComparisonDescriptor& descriptor);

class NeonComparisonWorkload : public NeonBaseWorkload<ComparisonQueueDescriptor>
{
public:
    NeonComparisonWorkload(const ComparisonQueueDescriptor& descriptor, const WorkloadInfo& info);

    virtual void Execute() const override;

private:
    mutable arm_compute::NEElementwiseComparison m_ComparisonLayer;
};

} //namespace armnn