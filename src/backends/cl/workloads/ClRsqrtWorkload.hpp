//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/CL/functions/CLElementWiseUnaryLayer.h>

namespace armnn
{

arm_compute::Status ClRsqrtWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class ClRsqrtWorkload : public BaseWorkload<RsqrtQueueDescriptor>
{
public:
    ClRsqrtWorkload(const RsqrtQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLRsqrtLayer m_RsqrtLayer;
};

} // namespace armnn
