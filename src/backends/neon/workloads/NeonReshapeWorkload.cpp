//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonReshapeWorkload.hpp"

namespace armnn
{

NeonReshapeWorkload::NeonReshapeWorkload(const ReshapeQueueDescriptor& descriptor,
                                         const WorkloadInfo& info)
    : BaseWorkload<ReshapeQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonReshapeWorkload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_Layer.configure(&input, &output);
}

void NeonReshapeWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonReshapeWorkload_Execute");
    m_Layer.run();
}

} //namespace armnn
