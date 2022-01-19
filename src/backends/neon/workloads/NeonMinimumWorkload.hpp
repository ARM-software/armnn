//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <neon/workloads/NeonWorkloadUtils.hpp>

#include <arm_compute/runtime/NEON/functions/NEElementwiseOperations.h>

namespace armnn
{

/// Validate function for validating the inputs and output.
/// @param [in] input0 The input0 value to be validated.
/// @param [in] input1 The input1 value to be validated.
/// @param [in] output The output value to be validated.
arm_compute::Status NeonMinimumWorkloadValidate(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output);

class NeonMinimumWorkload : public NeonBaseWorkload<MinimumQueueDescriptor>
{
public:
    /// Create a NeonMinimumWorkload.
    /// @param [in] descriptor The MinimumQueueDescriptor to configure this operation.
    /// @param [in] info The workload where this operation can be found.
    NeonMinimumWorkload(const MinimumQueueDescriptor& descriptor, const WorkloadInfo& info);

    /// Execute the Minimum operation.
    virtual void Execute() const override;

private:
    mutable arm_compute::NEElementwiseMin m_MinLayer;
};

} //namespace armnn
