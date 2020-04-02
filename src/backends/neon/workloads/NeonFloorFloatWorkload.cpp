//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonFloorFloatWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <arm_compute/runtime/NEON/functions/NEFloor.h>

namespace armnn
{
NeonFloorFloatWorkload::NeonFloorFloatWorkload(const FloorQueueDescriptor& descriptor,
                                               const WorkloadInfo& info)
    : FloatWorkload<FloorQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonFloorFloatWorkload", 1, 1);

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    auto layer = std::make_unique<arm_compute::NEFloor>();
    layer->configure(&input, &output);
    m_Layer.reset(layer.release());
}

void NeonFloorFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonFloorFloatWorkload_Execute");
    m_Layer->run();
}
} //namespace armnn



