//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <arm_compute/core/Error.h>
#include <backendsCommon/Workload.hpp>

#include <arm_compute/core/CL/CLCompileContext.h>

namespace armnn
{
arm_compute::Status ClConstantWorkloadValidate(const TensorInfo& output);

class ClConstantWorkload : public BaseWorkload<ConstantQueueDescriptor>
{
public:
    ClConstantWorkload(const ConstantQueueDescriptor& descriptor,
                       const WorkloadInfo& info,
                       const arm_compute::CLCompileContext& clCompileContext);

    void Execute() const override;

private:
    mutable bool m_RanOnce;
};

} //namespace armnn
