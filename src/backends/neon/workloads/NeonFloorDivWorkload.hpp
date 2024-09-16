//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/NEON/functions/NEFloor.h"
#include "arm_compute/runtime/NEON/functions/NEElementwiseOperations.h"
#include "arm_compute/runtime/NEON/functions/NECast.h"

namespace armnn
{



/// Validation for the Floor Div Workload
arm_compute::Status NeonFloorDivWorkloadValidate(const TensorInfo& input0Info,
                                                 const TensorInfo& input1Info,
                                                 const TensorInfo& outputInfo,
                                                 const ActivationDescriptor* activationDescriptor);

class NeonFloorDivWorkload : public NeonBaseWorkload<DivisionQueueDescriptor>
{
public:
    NeonFloorDivWorkload(const DivisionQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    arm_compute::Tensor m_OutputCast0;
    arm_compute::Tensor m_OutputCast1;
    arm_compute::Tensor m_OutputDiv;
    arm_compute::Tensor m_OutputFloor;
    arm_compute::Tensor m_OutputCast2;

    /// Cast layers only used for Signed32 types
    mutable std::unique_ptr<arm_compute::NECast> m_CastLayer0;
    mutable std::unique_ptr<arm_compute::NECast> m_CastLayer1;
    mutable std::unique_ptr<arm_compute::NECast> m_CastLayer2;

    mutable arm_compute::NEElementwiseDivision m_DivLayer;
    mutable arm_compute::NEFloor m_FloorLayer;
};

} //namespace armnn
