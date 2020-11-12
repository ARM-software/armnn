//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLElementwiseOperations.h>

namespace armnn
{

arm_compute::Status ClDivisionWorkloadValidate(const TensorInfo& input0,
                                               const TensorInfo& input1,
                                               const TensorInfo& output,
                                               const ActivationDescriptor* activationDescriptor = nullptr);

class ClDivisionFloatWorkload : public FloatWorkload<DivisionQueueDescriptor>
{
public:
    ClDivisionFloatWorkload(const DivisionQueueDescriptor& descriptor, const
    WorkloadInfo& info);

    using FloatWorkload<DivisionQueueDescriptor>::FloatWorkload;
    void Execute() const override;

private:
    mutable arm_compute::CLArithmeticDivision m_ArithmeticDivision;
};

} //namespace armnn
