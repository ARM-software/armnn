//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

class ClFloorFloatWorkload : public FloatWorkload<FloorQueueDescriptor>
{
public:
    ClFloorFloatWorkload(const FloorQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::CLFloor m_Layer;
};

} //namespace armnn




