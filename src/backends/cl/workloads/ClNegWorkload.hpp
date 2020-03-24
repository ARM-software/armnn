//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/CL/functions/CLElementWiseUnaryLayer.h>

namespace armnn
{

arm_compute::Status ClNegWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class ClNegWorkload : public BaseWorkload<ElementwiseUnaryQueueDescriptor>
{
public:
    ClNegWorkload(const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLNegLayer m_NegLayer;
};

} // namespace armnn
