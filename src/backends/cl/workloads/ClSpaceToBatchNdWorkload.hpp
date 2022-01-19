//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>
#include <armnn/Descriptors.hpp>

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLSpaceToBatchLayer.h>

namespace armnn
{

arm_compute::Status ClSpaceToBatchNdWorkloadValidate(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const SpaceToBatchNdDescriptor& descriptor);

class ClSpaceToBatchNdWorkload : public ClBaseWorkload<SpaceToBatchNdQueueDescriptor>
{
public:
    ClSpaceToBatchNdWorkload(const SpaceToBatchNdQueueDescriptor& descriptor,
                             const WorkloadInfo& info,
                             const arm_compute::CLCompileContext& clCompileContext);
    void Execute() const override;

private:
    mutable arm_compute::CLSpaceToBatchLayer m_SpaceToBatchLayer;
};

} //namespace armnn

