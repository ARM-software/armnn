//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonBatchNormalizationWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <armnn/backends/TensorHandle.hpp>

#include <arm_compute/runtime/NEON/functions/NEBatchNormalizationLayer.h>

namespace armnn
{
using namespace armcomputetensorutils;


arm_compute::Status NeonBatchNormalizationValidate(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const TensorInfo& mean,
                                                   const TensorInfo& var,
                                                   const TensorInfo& beta,
                                                   const TensorInfo& gamma,
                                                   const BatchNormalizationDescriptor& descriptor,
                                                   const ActivationDescriptor* activationDescriptor)
{
    const arm_compute::TensorInfo aclInputInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclMeanInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(mean, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclVarInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(var, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclBetaInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(beta, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclGammaInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(gamma, descriptor.m_DataLayout);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertActivationDescriptorToAclActivationLayerInfo(
            activationDescriptor);

    return arm_compute::NEBatchNormalizationLayer::validate(&aclInputInfo,
                                                            &aclOutputInfo,
                                                            &aclMeanInfo,
                                                            &aclVarInfo,
                                                            &aclBetaInfo,
                                                            &aclGammaInfo,
                                                            descriptor.m_Eps,
                                                            activationInfo);
}

NeonBatchNormalizationWorkload::NeonBatchNormalizationWorkload(
    const BatchNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info)
    : NeonBaseWorkload<BatchNormalizationQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonBatchNormalizationWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("NeonBatchNormalizationWorkload", 1, 1);

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    m_Mean = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_Mean, m_Data.m_Mean->GetTensorInfo());

    m_Variance = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_Variance, m_Data.m_Variance->GetTensorInfo());

    m_Gamma = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_Gamma, m_Data.m_Gamma->GetTensorInfo());

    m_Beta = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_Beta, m_Data.m_Beta->GetTensorInfo());

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    auto layer = std::make_unique<arm_compute::NEBatchNormalizationLayer>();
    layer->configure(&input,
                     &output,
                     m_Mean.get(),
                     m_Variance.get(),
                     m_Beta.get(),
                     m_Gamma.get(),
                     m_Data.m_Parameters.m_Eps,
                     activationInfo);
    m_Layer.reset(layer.release());

    InitializeArmComputeTensorData(*m_Mean, m_Data.m_Mean);
    InitializeArmComputeTensorData(*m_Variance, m_Data.m_Variance);
    InitializeArmComputeTensorData(*m_Gamma, m_Data.m_Gamma);
    InitializeArmComputeTensorData(*m_Beta, m_Data.m_Beta);

    // Force Compute Library to perform the necessary copying and reshaping, after which
    // delete all the input tensors that will no longer be needed
    m_Layer->prepare();
    FreeUnusedTensors();
}

void NeonBatchNormalizationWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonBatchNormalizationWorkload_Execute", this->GetGuid());
    m_Layer->run();
}

void NeonBatchNormalizationWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_Mean);
    FreeTensorIfUnused(m_Variance);
    FreeTensorIfUnused(m_Gamma);
    FreeTensorIfUnused(m_Beta);
}

} //namespace armnn
