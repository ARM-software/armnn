//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonSoftmaxUint8Workload.hpp"

namespace armnn
{

NeonSoftmaxUint8Workload::NeonSoftmaxUint8Workload(const SoftmaxQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info,
                                                   std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : Uint8Workload<SoftmaxQueueDescriptor>(descriptor, info)
    , m_SoftmaxLayer(memoryManager)
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

    m_SoftmaxLayer.configure(&input, &output, descriptor.m_Parameters.m_Beta);
}

void NeonSoftmaxUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "ClSoftmaxUint8Workload_Execute");

    m_SoftmaxLayer.run();
}

} //namespace armnn

