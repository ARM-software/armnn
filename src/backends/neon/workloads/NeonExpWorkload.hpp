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

arm_compute::Status NeonExpWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class NeonExpWorkload : public NeonBaseWorkload<ElementwiseUnaryQueueDescriptor>
{
public:
    NeonExpWorkload(const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEExpLayer m_ExpLayer;
};

} //namespace armnn





