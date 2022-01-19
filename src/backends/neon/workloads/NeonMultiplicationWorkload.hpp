//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/IFunction.h>

#include <memory>

namespace armnn
{
arm_compute::Status NeonMultiplicationWorkloadValidate(const TensorInfo& input0,
                                                       const TensorInfo& input1,
                                                       const TensorInfo& output,
                                                       const ActivationDescriptor* activationDescriptor = nullptr);

class NeonMultiplicationWorkload : public NeonBaseWorkload<MultiplicationQueueDescriptor>
{
public:
    NeonMultiplicationWorkload(const MultiplicationQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_PixelWiseMultiplication;
};

} //namespace armnn
