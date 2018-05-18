//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonL2NormalizationFloat32Workload.hpp"
#include "backends/ArmComputeUtils.hpp"

namespace armnn
{

NeonL2NormalizationFloat32Workload::NeonL2NormalizationFloat32Workload(const L2NormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info, std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : Float32Workload<L2NormalizationQueueDescriptor>(descriptor, info)
    , m_Layer(memoryManager)
{
    m_Data.ValidateInputsOutputs("NeonL2NormalizationFloat32Workload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    m_Layer.configure(&input, &output, CreateAclNormalizationLayerInfoForL2Normalization(info.m_InputTensorInfos[0]));
}

void NeonL2NormalizationFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "NeonL2NormalizationFloat32Workload_Execute");
    m_Layer.run();
}

} //namespace armnn
