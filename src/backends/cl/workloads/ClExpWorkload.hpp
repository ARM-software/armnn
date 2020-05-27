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

arm_compute::Status ClExpWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class ClExpWorkload : public BaseWorkload<ElementwiseUnaryQueueDescriptor>
{
public:
    ClExpWorkload(const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLExpLayer m_ExpLayer;
};

} // namespace armnn
