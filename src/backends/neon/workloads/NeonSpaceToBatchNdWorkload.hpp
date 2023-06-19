//
// Copyright © 2020-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"
#include "NeonWorkloadUtils.hpp"

#include <arm_compute/runtime/NEON/functions/NESpaceToBatchLayer.h>
#include <arm_compute/runtime/NEON/functions/NEReshapeLayer.h>

namespace armnn
{

arm_compute::Status NeonSpaceToBatchNdWorkloadValidate(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const SpaceToBatchNdDescriptor& descriptor);

class NeonSpaceToBatchNdWorkload : public NeonBaseWorkload<SpaceToBatchNdQueueDescriptor>
{
public:
    using NeonBaseWorkload<SpaceToBatchNdQueueDescriptor>::NeonBaseWorkload;

    NeonSpaceToBatchNdWorkload(const SpaceToBatchNdQueueDescriptor& descriptor, const WorkloadInfo& info);

    virtual void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::NESpaceToBatchLayer> m_Layer;
    mutable std::unique_ptr<arm_compute::NEReshapeLayer> m_LayerReshapeInput;
    mutable std::unique_ptr<arm_compute::NEReshapeLayer> m_LayerReshapeOutput;
    arm_compute::Tensor m_ReshapeInputTensor;
    arm_compute::Tensor m_ReshapeOutputTensor;
};

} //namespace armnn