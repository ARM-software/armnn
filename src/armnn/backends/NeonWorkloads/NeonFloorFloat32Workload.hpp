//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

namespace armnn
{

class NeonFloorFloat32Workload : public FloatWorkload<FloorQueueDescriptor>
{
public:
    NeonFloorFloat32Workload(const FloorQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEFloor m_Layer;
};

} //namespace armnn




