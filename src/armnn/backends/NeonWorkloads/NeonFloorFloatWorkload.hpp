//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>

namespace armnn
{

class NeonFloorFloatWorkload : public FloatWorkload<FloorQueueDescriptor>
{
public:
    NeonFloorFloatWorkload(const FloorQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEFloor m_Layer;
};

} //namespace armnn




