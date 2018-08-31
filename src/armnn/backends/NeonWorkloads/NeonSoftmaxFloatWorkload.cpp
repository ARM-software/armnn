//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonSoftmaxFloatWorkload.hpp"

namespace armnn
{

NeonSoftmaxFloatWorkload::NeonSoftmaxFloatWorkload(const SoftmaxQueueDescriptor& descriptor,
    const WorkloadInfo& info, std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : FloatWorkload<SoftmaxQueueDescriptor>(descriptor, info)
    , m_SoftmaxLayer(memoryManager)
{
    m_Data.ValidateInputsOutputs("NeonSoftmaxFloatWorkload", 1, 1);

    // The ArmCompute softmax layer uses 2D input/output tensors, so flatten the first three dimensions.
    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_SoftmaxLayer.configure(&input, &output, m_Data.m_Parameters.m_Beta);
}

void NeonSoftmaxFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonSoftmaxFloatWorkload_Execute");
    m_SoftmaxLayer.run();
}

} //namespace armnn

