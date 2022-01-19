//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLElementwiseOperations.h>

namespace armnn
{

class ClSubtractionWorkload : public ClBaseWorkload<SubtractionQueueDescriptor>
{
public:
    ClSubtractionWorkload(const SubtractionQueueDescriptor& descriptor,
                          const WorkloadInfo& info,
                          const arm_compute::CLCompileContext& clCompileContext);

    void Execute() const override;

private:
    mutable arm_compute::CLArithmeticSubtraction m_Layer;
};

arm_compute::Status ClSubtractionValidate(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          const ActivationDescriptor* activationDescriptor = nullptr);
} //namespace armnn
