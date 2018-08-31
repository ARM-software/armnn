//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonReshapeFloatWorkload.hpp"



namespace armnn
{

NeonReshapeFloatWorkload::NeonReshapeFloatWorkload(const ReshapeQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info)
    : FloatWorkload<ReshapeQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonReshapeFloatWorkload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_Layer.configure(&input, &output);
}

void NeonReshapeFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonReshapeFloatWorkload_Execute");
    m_Layer.run();
}

} //namespace armnn

