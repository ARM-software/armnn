//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonNormalizationFloatWorkload.hpp"

#include "NeonWorkloadUtils.hpp"
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <arm_compute/runtime/NEON/functions/NENormalizationLayer.h>

using namespace armnn::armcomputetensorutils;

namespace armnn
{

namespace
{
using ACLMemManagerOnDemand = std::shared_ptr<arm_compute::MemoryManagerOnDemand>;

bool IsNeonNormalizationDescriptorSupported(const NormalizationDescriptor& parameters,
                                            Optional<std::string&> reasonIfUnsupported)
{
    if (parameters.m_NormMethodType != NormalizationAlgorithmMethod::LocalBrightness)
    {
        if (reasonIfUnsupported)
        {
            reasonIfUnsupported.value() = "Unsupported normalisation method type, only LocalBrightness is supported";
        }
        return false;
    }
    if (parameters.m_NormSize % 2 == 0)
    {
        if (reasonIfUnsupported)
        {
            reasonIfUnsupported.value() = "Normalization size must be an odd number.";
        }
        return false;
    }

    return true;
}

} // anonymous namespace

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
                                                               ACLMemManagerOnDemand& memoryManager)
    : FloatWorkload<NormalizationQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonNormalizationWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("NeonNormalizationFloatWorkload", 1, 1);
    std::string reasonIfUnsupported;
    if (!IsNeonNormalizationDescriptorSupported(m_Data.m_Parameters, Optional<std::string&>(reasonIfUnsupported)))
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

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    const arm_compute::NormType normType =
        ConvertNormalizationAlgorithmChannelToAclNormType(m_Data.m_Parameters.m_NormChannelType);
    arm_compute::NormalizationLayerInfo normalizationInfo(normType,
                                                          m_Data.m_Parameters.m_NormSize,
                                                          m_Data.m_Parameters.m_Alpha,
                                                          m_Data.m_Parameters.m_Beta,
                                                          m_Data.m_Parameters.m_K,
                                                          false);
    auto layer = std::make_unique<arm_compute::NENormalizationLayer>(memoryManager);
    layer->configure(&input, &output, normalizationInfo);
    m_NormalizationLayer.reset(layer.release());
}

void NeonNormalizationFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonNormalizationFloatWorkload_Execute", this->GetGuid());
    m_NormalizationLayer->run();
}

void NeonNormalizationFloatWorkload::ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot)
{
    ITensorHandle* backupHandle = this->m_Data.m_Inputs[slot];
    this->m_Data.m_Inputs[slot] = tensorHandle;
    try
    {
        Reconfigure();
    }
    catch(armnn::UnimplementedException& e)
    {
        // Cannot reconfigure, revert the slot back and throw the exception.
        this->m_Data.m_Inputs[slot] = backupHandle;
        throw e;
    }
}

// Replace output tensor handle with the given TensorHandle
void NeonNormalizationFloatWorkload::ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot)
{
    ITensorHandle* backupHandle = this->m_Data.m_Inputs[slot];
    this->m_Data.m_Inputs[slot] = tensorHandle;
    try
    {
        Reconfigure();
    }
    catch(armnn::UnimplementedException& e)
    {
        // Cannot reconfigure, revert the slot back and throw the exception.
        this->m_Data.m_Inputs[slot] = backupHandle;
        throw e;
    }
}

void NeonNormalizationFloatWorkload::Reconfigure()
{
    throw armnn::UnimplementedException("Reconfigure not implemented for this workload");
}

} //namespace armnn
