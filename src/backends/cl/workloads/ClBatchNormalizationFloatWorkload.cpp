//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClBatchNormalizationFloatWorkload.hpp"
#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>
#include <cl/ClTensorHandle.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClBatchNormalizationValidate(const TensorInfo& input,
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

    return arm_compute::CLBatchNormalizationLayer::validate(&aclInputInfo,
                                                            &aclOutputInfo,
                                                            &aclMeanInfo,
                                                            &aclVarInfo,
                                                            &aclBetaInfo,
                                                            &aclGammaInfo,
                                                            descriptor.m_Eps,
                                                            activationInfo);
}

ClBatchNormalizationFloatWorkload::ClBatchNormalizationFloatWorkload(
    const BatchNormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info,
    const arm_compute::CLCompileContext& clCompileContext)
    : FloatWorkload<BatchNormalizationQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClBatchNormalizationWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Mean = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_Mean, m_Data.m_Mean->GetTensorInfo());

    m_Variance = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_Variance, m_Data.m_Variance->GetTensorInfo());

    m_Gamma = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_Gamma, m_Data.m_Gamma->GetTensorInfo());

    m_Beta = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_Beta, m_Data.m_Beta->GetTensorInfo());

    m_Data.ValidateInputsOutputs("ClBatchNormalizationFloatWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClBatchNormalizationFloatWorkload_configure");
        m_Layer.configure(clCompileContext,
                          &input,
                          &output,
                          m_Mean.get(),
                          m_Variance.get(),
                          m_Beta.get(),
                          m_Gamma.get(),
                          m_Data.m_Parameters.m_Eps,
                          activationInfo);
    }

    InitializeArmComputeClTensorData(*m_Mean, m_Data.m_Mean);
    InitializeArmComputeClTensorData(*m_Variance, m_Data.m_Variance);
    InitializeArmComputeClTensorData(*m_Beta, m_Data.m_Beta);
    InitializeArmComputeClTensorData(*m_Gamma, m_Data.m_Gamma);

    // Force Compute Library to perform the necessary copying and reshaping, after which
    // delete all the input tensors that will no longer be needed
    m_Layer.prepare();
    FreeUnusedTensors();
}

void ClBatchNormalizationFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClBatchNormalizationFloatWorkload_Execute", this->GetGuid());
    RunClFunction(m_Layer, CHECK_LOCATION());
}

void ClBatchNormalizationFloatWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_Mean);
    FreeTensorIfUnused(m_Variance);
    FreeTensorIfUnused(m_Gamma);
    FreeTensorIfUnused(m_Beta);
}

void ClBatchNormalizationFloatWorkload::ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot)
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
void ClBatchNormalizationFloatWorkload::ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot)
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

void ClBatchNormalizationFloatWorkload::Reconfigure()
{
    throw armnn::UnimplementedException("Reconfigure not implemented for this workload");
}

} //namespace armnn
