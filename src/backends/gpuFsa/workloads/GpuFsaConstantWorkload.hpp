//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "GpuFsaBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/core/CL/CLCompileContext.h>

namespace armnn
{
    arm_compute::Status GpuFsaConstantWorkloadValidate(const TensorInfo& output);

    class GpuFsaConstantWorkload : public GpuFsaBaseWorkload<ConstantQueueDescriptor>
    {
    public:
        GpuFsaConstantWorkload(const ConstantQueueDescriptor& descriptor,
                               const WorkloadInfo& info,
                               const arm_compute::CLCompileContext& clCompileContext);

        void Execute() const override;

    private:
        mutable bool m_RanOnce;
    };

} //namespace armnn
