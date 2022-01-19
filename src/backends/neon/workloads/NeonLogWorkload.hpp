//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"
#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/NEON/functions/NEElementwiseUnaryLayer.h>

namespace armnn
{

arm_compute::Status NeonLogWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class NeonLogWorkload : public NeonBaseWorkload<ElementwiseUnaryQueueDescriptor>
{
public:
    NeonLogWorkload(const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NELogLayer m_LogLayer;
};

} //namespace armnn