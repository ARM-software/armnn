//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

class ClFloorFloat32Workload : public FloatWorkload<FloorQueueDescriptor>
{
public:
    ClFloorFloat32Workload(const FloorQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::CLFloor m_Layer;
};

} //namespace armnn




