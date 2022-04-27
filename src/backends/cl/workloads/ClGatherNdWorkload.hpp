//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/CL/functions/CLGather.h"
#include "arm_compute/runtime/CL/functions/CLPixelWiseMultiplication.h"
#include "arm_compute/runtime/CL/functions/CLReductionOperation.h"
#include "arm_compute/runtime/CL/functions/CLReshapeLayer.h"

namespace armnn
{
arm_compute::Status ClGatherNdWorkloadValidate(const TensorInfo& params,
                                               const TensorInfo& indices,
                                               const TensorInfo& output);

class ClGatherNdWorkload : public ClBaseWorkload<GatherNdQueueDescriptor>
{
public:
    ClGatherNdWorkload(const GatherNdQueueDescriptor& descriptor,
                       const WorkloadInfo& info,
                       const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

private:
    arm_compute::CLTensor m_FlattenedCoeff;
    arm_compute::CLTensor m_OutputMul;
    arm_compute::CLTensor m_FlattenedIndices;
    arm_compute::CLTensor m_OutputGather;

    mutable arm_compute::CLPixelWiseMultiplication m_MulLayer;
    mutable arm_compute::CLReductionOperation m_ReduceSumLayer;
    mutable arm_compute::CLGather m_GatherLayer;
    mutable arm_compute::CLReshapeLayer m_ReshapeLayer;
};

} //namespace armnn