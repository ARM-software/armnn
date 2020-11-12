//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLElementwiseOperations.h>

namespace armnn
{

class ClAdditionWorkload : public BaseWorkload<AdditionQueueDescriptor>
{
public:
    ClAdditionWorkload(const AdditionQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::CLArithmeticAddition m_Layer;
};

arm_compute::Status ClAdditionValidate(const TensorInfo& input0,
                                       const TensorInfo& input1,
                                       const TensorInfo& output,
                                       const ActivationDescriptor* activationDescriptor = nullptr);
} //namespace armnn
