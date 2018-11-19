//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonL2NormalizationFloatWorkload.hpp"
#include <aclCommon/ArmComputeUtils.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status NeonL2NormalizationWorkloadValidate(const TensorInfo& input,
                                                        const TensorInfo& output,
                                                        const L2NormalizationDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInput = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutput = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);

    arm_compute::NormalizationLayerInfo normalizationInfo =
            CreateAclNormalizationLayerInfoForL2Normalization(input, descriptor.m_DataLayout);

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

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    m_Layer.configure(&input,
                      &output,
                      CreateAclNormalizationLayerInfoForL2Normalization(
                          info.m_InputTensorInfos[0], m_Data.m_Parameters.m_DataLayout));
}

void NeonL2NormalizationFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonL2NormalizationFloatWorkload_Execute");
    m_Layer.run();
}

} //namespace armnn
