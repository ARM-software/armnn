//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/NEON/functions/NEGather.h"
#include "arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h"
#include "arm_compute/runtime/NEON/functions/NEReductionOperation.h"
#include "arm_compute/runtime/NEON/functions/NEReshapeLayer.h"

namespace armnn
{
arm_compute::Status NeonGatherNdWorkloadValidate(const TensorInfo& input,
                                                 const TensorInfo& indices,
                                                 const TensorInfo& output);

class NeonGatherNdWorkload : public NeonBaseWorkload<GatherNdQueueDescriptor>
{
public:
    NeonGatherNdWorkload(const GatherNdQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    arm_compute::Tensor m_FlattenedCoeff;
    arm_compute::Tensor m_OutputMul;
    arm_compute::Tensor m_FlattenedIndices;
    arm_compute::Tensor m_OutputGather;

    mutable arm_compute::NEPixelWiseMultiplication m_MulLayer;
    mutable arm_compute::NEReductionOperation m_ReduceSumLayer;
    mutable arm_compute::NEGather m_GatherLayer;
    mutable arm_compute::NEReshapeLayer m_ReshapeLayer;

};

} //namespace armnn