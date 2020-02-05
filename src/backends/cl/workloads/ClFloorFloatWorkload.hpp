//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLFloor.h>

namespace armnn
{

arm_compute::Status ClFloorWorkloadValidate(const TensorInfo& input,
                                            const TensorInfo& output);

class ClFloorFloatWorkload : public FloatWorkload<FloorQueueDescriptor>
{
public:
    ClFloorFloatWorkload(const FloorQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::CLFloor m_Layer;
};

} //namespace armnn




