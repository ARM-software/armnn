//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonSoftmaxUint8Workload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <arm_compute/runtime/NEON/functions/NESoftmaxLayer.h>

namespace armnn
{

NeonSoftmaxUint8Workload::NeonSoftmaxUint8Workload(const SoftmaxQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info,
                                                   std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : Uint8Workload<SoftmaxQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonSoftmaxUint8Workload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    const auto outputQuantization = output.info()->quantization_info();

    if ((outputQuantization.scale != (1.0f / 256.0f)) || (outputQuantization.offset != 0))
    {
        throw InvalidArgumentException(
            "Invalid quantization for output. Only scale = 1.0f / 256.0f and offset = 0 supported");
    }

    auto layer = std::make_unique<arm_compute::NESoftmaxLayer>(memoryManager);
    layer->configure(&input, &output, descriptor.m_Parameters.m_Beta);
    m_SoftmaxLayer.reset(layer.release());
}

void NeonSoftmaxUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonSoftmaxUint8Workload_Execute");

    m_SoftmaxLayer->run();
}

} //namespace armnn

