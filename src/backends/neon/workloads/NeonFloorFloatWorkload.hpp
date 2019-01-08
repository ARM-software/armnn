//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/Tensor.h>

#include <memory>

namespace armnn
{

class NeonFloorFloatWorkload : public FloatWorkload<FloorQueueDescriptor>
{
public:
    NeonFloorFloatWorkload(const FloorQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_Layer;
};

} //namespace armnn




