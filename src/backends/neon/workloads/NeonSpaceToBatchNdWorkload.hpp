//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>
#include <armnn/Descriptors.hpp>

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/NEON/functions/NESpaceToBatchLayer.h>

namespace armnn
{

arm_compute::Status NeonSpaceToBatchNdWorkloadValidate(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const SpaceToBatchNdDescriptor& descriptor);

class NeonSpaceToBatchNdWorkload : public BaseWorkload<SpaceToBatchNdQueueDescriptor>
{
public:
    using BaseWorkload<SpaceToBatchNdQueueDescriptor>::BaseWorkload;

    NeonSpaceToBatchNdWorkload(const SpaceToBatchNdQueueDescriptor& descriptor, const WorkloadInfo& info);

    virtual void Execute() const override;

private:
    mutable std::unique_ptr<arm_compute::NESpaceToBatchLayer> m_Layer;
};

} //namespace armnn