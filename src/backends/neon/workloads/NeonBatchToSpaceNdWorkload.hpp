//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>
#include <armnn/Descriptors.hpp>

#include "NeonBaseWorkload.hpp"

#include <arm_compute/runtime/NEON/functions/NEBatchToSpaceLayer.h>

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
};

}
