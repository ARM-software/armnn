//
// Copyright © 2020-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"
#include "NeonWorkloadUtils.hpp"

#include <arm_compute/runtime/NEON/functions/NEBatchToSpaceLayer.h>
#include <arm_compute/runtime/NEON/functions/NEReshapeLayer.h>

namespace armnn
{

arm_compute::Status NeonBatchToSpaceNdWorkloadValidate(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const BatchToSpaceNdDescriptor& descriptor);

class NeonBatchToSpaceNdWorkload : public NeonBaseWorkload<BatchToSpaceNdQueueDescriptor>
{
public:
    using NeonBaseWorkload<BatchToSpaceNdQueueDescriptor>::NeonBaseWorkload;

    NeonBatchToSpaceNdWorkload(const BatchToSpaceNdQueueDescriptor& descriptor, const WorkloadInfo& info);

    virtual void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::NEBatchToSpaceLayer> m_Layer;
    mutable std::unique_ptr<arm_compute::NEReshapeLayer> m_LayerReshapeInput;
    mutable std::unique_ptr<arm_compute::NEReshapeLayer> m_LayerReshapeOutput;
    arm_compute::Tensor m_ReshapeInputTensor;
    arm_compute::Tensor m_ReshapeOutputTensor;
};

}
