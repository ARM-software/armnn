//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLPixelWiseMultiplication.h>

namespace armnn
{

arm_compute::Status ClMultiplicationWorkloadValidate(const TensorInfo& input0,
                                                     const TensorInfo& input1,
                                                     const TensorInfo& output,
                                                     const ActivationDescriptor* activationDescriptor = nullptr);

class ClMultiplicationWorkload : public ClBaseWorkload<MultiplicationQueueDescriptor>
{
public:
    ClMultiplicationWorkload(const MultiplicationQueueDescriptor& descriptor,
                             const WorkloadInfo& info,
                             const arm_compute::CLCompileContext& clCompileContext);

    using ClBaseWorkload<MultiplicationQueueDescriptor>::ClBaseWorkload;
    void Execute() const override;

private:
    mutable arm_compute::CLPixelWiseMultiplication   m_PixelWiseMultiplication;
};

} //namespace armnn
