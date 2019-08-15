//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonSoftmaxFloatWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeUtils.hpp>
#include <arm_compute/runtime/NEON/functions/NESoftmaxLayer.h>

namespace armnn
{

NeonSoftmaxFloatWorkload::NeonSoftmaxFloatWorkload(const SoftmaxQueueDescriptor& descriptor,
    const WorkloadInfo& info, std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : FloatWorkload<SoftmaxQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonSoftmaxFloatWorkload", 1, 1);

    // The ArmCompute softmax layer uses 2D input/output tensors, so flatten the first three dimensions.
    arm_compute::ITensor& input = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    auto layer = std::make_unique<arm_compute::NESoftmaxLayer>(memoryManager);
    unsigned int aclAxis = ComputeSoftmaxAclAxis(m_Data.m_Parameters, info.m_InputTensorInfos[0]);
    layer->configure(&input, &output, m_Data.m_Parameters.m_Beta, aclAxis);
    m_SoftmaxLayer.reset(layer.release());
}

void NeonSoftmaxFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonSoftmaxFloatWorkload_Execute");
    m_SoftmaxLayer->run();
}

} //namespace armnn

