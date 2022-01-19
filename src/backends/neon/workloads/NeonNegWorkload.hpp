//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/NEON/functions/NEElementwiseUnaryLayer.h>

namespace armnn
{

arm_compute::Status NeonNegWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class NeonNegWorkload : public NeonBaseWorkload<ElementwiseUnaryQueueDescriptor>
{
public:
    NeonNegWorkload(const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NENegLayer m_NegLayer;
};

} // namespace armnn
