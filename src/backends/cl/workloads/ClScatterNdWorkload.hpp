//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Descriptors.hpp>

#include <arm_compute/runtime/CL/functions/CLScatter.h>

#include "ClBaseWorkload.hpp"

namespace armnn
{

arm_compute::Status ClScatterNdWorkloadValidate(const TensorInfo& input,
                                                const TensorInfo& indices,
                                                const TensorInfo& updates,
                                                const TensorInfo& output,
                                                const ScatterNdDescriptor& descriptor);

class ClScatterNdWorkload : public ClBaseWorkload<ScatterNdQueueDescriptor>
{
public:
    ClScatterNdWorkload(const ScatterNdQueueDescriptor& descriptor,
                        const WorkloadInfo& info,
                        const arm_compute::CLCompileContext& clCompileContext);
    void Execute() const override;

private:
    mutable arm_compute::CLScatter m_ScatterNdLayer;
};

} //namespace armnn
