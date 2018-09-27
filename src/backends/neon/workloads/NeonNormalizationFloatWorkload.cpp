//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonNormalizationFloatWorkload.hpp"
#include <backends/neon/NeonLayerSupport.hpp>
#include <backends/aclCommon/ArmComputeUtils.hpp>
#include <backends/aclCommon/ArmComputeTensorUtils.hpp>

using namespace armnn::armcomputetensorutils;

namespace armnn
{

arm_compute::Status NeonNormalizationWorkloadValidate(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const NormalizationDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInput = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutput = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);

    arm_compute::NormalizationLayerInfo normalizationInfo = BuildArmComputeNormalizationLayerInfo(descriptor);

    return arm_compute::NENormalizationLayer::validate(&aclInput, &aclOutput, normalizationInfo);
}

NeonNormalizationFloatWorkload::NeonNormalizationFloatWorkload(const NormalizationQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info,
                                                   std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : FloatWorkload<NormalizationQueueDescriptor>(descriptor, info)
    , m_NormalizationLayer(memoryManager)
{
    m_Data.ValidateInputsOutputs("NeonNormalizationFloatWorkload", 1, 1);
    std::string reasonIfUnsupported;
    if (!IsNeonNormalizationDescParamsSupported(&reasonIfUnsupported, m_Data.m_Parameters))
    {
        throw UnimplementedException(reasonIfUnsupported);
    }

    // Input and output tensors have to have the same dimensionality.
    if (info.m_InputTensorInfos[0].GetShape()[1] != info.m_OutputTensorInfos[0].GetShape()[1]
        || info.m_InputTensorInfos[0].GetShape()[0] != info.m_OutputTensorInfos[0].GetShape()[0]
        || info.m_InputTensorInfos[0].GetShape()[3] != info.m_OutputTensorInfos[0].GetShape()[3]
        || info.m_InputTensorInfos[0].GetShape()[2] != info.m_OutputTensorInfos[0].GetShape()[2])
    {
        throw InvalidArgumentException("Normalization requires input and output tensors to have equal dimensionality.");
    }

    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    const arm_compute::NormType normType =
        ConvertNormalizationAlgorithmChannelToAclNormType(m_Data.m_Parameters.m_NormChannelType);
    arm_compute::NormalizationLayerInfo normalizationInfo(normType,
                                                          m_Data.m_Parameters.m_NormSize,
                                                          m_Data.m_Parameters.m_Alpha,
                                                          m_Data.m_Parameters.m_Beta,
                                                          m_Data.m_Parameters.m_K,
                                                          false);

    m_NormalizationLayer.configure(&input, &output, normalizationInfo);
}

void NeonNormalizationFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonNormalizationFloatWorkload_Execute");
    m_NormalizationLayer.run();
}

} //namespace armnn
