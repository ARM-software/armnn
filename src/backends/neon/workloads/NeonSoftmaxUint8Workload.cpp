//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonSoftmaxUint8Workload.hpp"
#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeUtils.hpp>

#include <arm_compute/runtime/NEON/functions/NESoftmaxLayer.h>

namespace armnn
{

NeonSoftmaxUint8Workload::NeonSoftmaxUint8Workload(const SoftmaxQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info,
                                                   std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : Uint8Workload<SoftmaxQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonSoftmaxUint8Workload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    const auto outputQuantization = output.info()->quantization_info();

    if ((!outputQuantization.scale().empty() && outputQuantization.scale()[0] != (1.0f / 256.0f)) ||
        (!outputQuantization.offset().empty() && outputQuantization.offset()[0] != 0) ||
         outputQuantization.scale().empty() || outputQuantization.offset().empty())
    {
        throw InvalidArgumentException(
            "Invalid quantization for output. Only scale = 1.0f / 256.0f and offset = 0 supported");
    }

    auto layer = std::make_unique<arm_compute::NESoftmaxLayer>(memoryManager);
    unsigned int aclAxis = ComputeSoftmaxAclAxis(m_Data.m_Parameters, info.m_InputTensorInfos[0]);
    layer->configure(&input, &output, descriptor.m_Parameters.m_Beta, aclAxis);
    m_SoftmaxLayer.reset(layer.release());
}

void NeonSoftmaxUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonSoftmaxUint8Workload_Execute");

    m_SoftmaxLayer->run();
}

} //namespace armnn

