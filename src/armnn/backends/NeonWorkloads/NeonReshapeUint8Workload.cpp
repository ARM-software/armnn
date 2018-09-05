//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonReshapeUint8Workload.hpp"




namespace armnn
{
NeonReshapeUint8Workload::NeonReshapeUint8Workload(const ReshapeQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info)
    : Uint8Workload<ReshapeQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonReshapeUint8Workload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_Layer.configure(&input, &output);
}

void NeonReshapeUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonReshapeUint8Workload_Execute");
    m_Layer.run();
}
} //namespace armnn
