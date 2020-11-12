//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLPixelWiseMultiplication.h>

namespace armnn
{

arm_compute::Status ClMultiplicationWorkloadValidate(const TensorInfo& input0,
                                                     const TensorInfo& input1,
                                                     const TensorInfo& output,
                                                     const ActivationDescriptor* activationDescriptor = nullptr);

class ClMultiplicationWorkload : public BaseWorkload<MultiplicationQueueDescriptor>
{
public:
    ClMultiplicationWorkload(const MultiplicationQueueDescriptor& descriptor, const WorkloadInfo& info);

    using BaseWorkload<MultiplicationQueueDescriptor>::BaseWorkload;
    void Execute() const override;

private:
    mutable arm_compute::CLPixelWiseMultiplication   m_PixelWiseMultiplication;
};

} //namespace armnn



