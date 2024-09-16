//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"
#include "ClWorkloadUtils.hpp"

#include <armnn/backends/Workload.hpp>
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/CL/functions/CLCast.h"
#include "arm_compute/runtime/CL/functions/CLElementwiseOperations.h"
#include "arm_compute/runtime/CL/functions/CLFloor.h"

namespace armnn
{
arm_compute::Status ClFloorDivWorkloadValidate(const TensorInfo& input0Info,
                                               const TensorInfo& input1Info,
                                               const TensorInfo& outputInfo,
                                               const ActivationDescriptor* activationDescriptor);

class ClFloorDivWorkload : public ClBaseWorkload<DivisionQueueDescriptor>
{
public:
    ClFloorDivWorkload(const DivisionQueueDescriptor& descriptor,
                       const WorkloadInfo& info,
                       const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

private:
    arm_compute::CLTensor m_OutputCast0;
    arm_compute::CLTensor m_OutputCast1;
    arm_compute::CLTensor m_OutputDiv;
    arm_compute::CLTensor m_OutputFloor;
    arm_compute::CLTensor m_OutputCast2;

    /// Cast layers only used for Signed32 types
    mutable std::unique_ptr<arm_compute::CLCast> m_CastLayer0;
    mutable std::unique_ptr<arm_compute::CLCast> m_CastLayer1;
    mutable std::unique_ptr<arm_compute::CLCast> m_CastLayer2;

    mutable arm_compute::CLArithmeticDivision m_DivLayer;
    mutable arm_compute::CLFloor m_FloorLayer;
};

} //namespace armnn