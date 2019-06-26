//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/TypesUtils.hpp>

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/NEON/functions/NESpaceToDepthLayer.h>

namespace armnn
{

arm_compute::Status NeonSpaceToDepthWorkloadValidate(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const SpaceToDepthDescriptor& descriptor);

class NeonSpaceToDepthWorkload : public BaseWorkload<SpaceToDepthQueueDescriptor>
{
public:
    using BaseWorkload<SpaceToDepthQueueDescriptor>::BaseWorkload;
    NeonSpaceToDepthWorkload(const SpaceToDepthQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;
private:
    mutable std::unique_ptr<arm_compute::NESpaceToDepthLayer> m_Layer;
};

} //namespace armnn