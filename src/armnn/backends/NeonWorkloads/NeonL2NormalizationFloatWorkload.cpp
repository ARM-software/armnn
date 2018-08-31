//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonL2NormalizationFloatWorkload.hpp"
#include "backends/ArmComputeUtils.hpp"

namespace armnn
{

arm_compute::Status NeonL2NormalizationWorkloadValidate(const TensorInfo& input,
                                                        const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    arm_compute::NormalizationLayerInfo normalizationInfo =
            CreateAclNormalizationLayerInfoForL2Normalization(input);

    return arm_compute::NENormalizationLayer::validate(&aclInput, &aclOutput, normalizationInfo);
}

NeonL2NormalizationFloatWorkload::NeonL2NormalizationFloatWorkload(const L2NormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info, std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : FloatWorkload<L2NormalizationQueueDescriptor>(descriptor, info)
    , m_Layer(memoryManager)
{
    m_Data.ValidateInputsOutputs("NeonL2NormalizationFloatWorkload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    m_Layer.configure(&input, &output, CreateAclNormalizationLayerInfoForL2Normalization(info.m_InputTensorInfos[0]));
}

void NeonL2NormalizationFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonL2NormalizationFloatWorkload_Execute");
    m_Layer.run();
}

} //namespace armnn
