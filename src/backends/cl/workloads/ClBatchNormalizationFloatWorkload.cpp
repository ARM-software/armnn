//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClBatchNormalizationFloatWorkload.hpp"
#include <cl/ClTensorHandle.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <cl/ClLayerSupport.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClBatchNormalizationValidate(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const TensorInfo& mean,
                                                 const TensorInfo& var,
                                                 const TensorInfo& beta,
                                                 const TensorInfo& gamma,
                                                 const BatchNormalizationDescriptor &desc)
{
    const arm_compute::TensorInfo aclInputInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(input, desc.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(output, desc.m_DataLayout);
    const arm_compute::TensorInfo aclMeanInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(mean, desc.m_DataLayout);
    const arm_compute::TensorInfo aclVarInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(var, desc.m_DataLayout);
    const arm_compute::TensorInfo aclBetaInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(beta, desc.m_DataLayout);
    const arm_compute::TensorInfo aclGammaInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(gamma, desc.m_DataLayout);

    return arm_compute::CLBatchNormalizationLayer::validate(&aclInputInfo,
                                                            &aclOutputInfo,
                                                            &aclMeanInfo,
                                                            &aclVarInfo,
                                                            &aclBetaInfo,
                                                            &aclGammaInfo,
                                                            desc.m_Eps);
}

ClBatchNormalizationFloatWorkload::ClBatchNormalizationFloatWorkload(
    const BatchNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info)
    : FloatWorkload<BatchNormalizationQueueDescriptor>(descriptor, info)
{
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

    m_Layer.configure(&input,
                      &output,
                      m_Mean.get(),
                      m_Variance.get(),
                      m_Beta.get(),
                      m_Gamma.get(),
                      m_Data.m_Parameters.m_Eps);

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
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClBatchNormalizationFloatWorkload_Execute");
    RunClFunction(m_Layer, CHECK_LOCATION());
}

void ClBatchNormalizationFloatWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_Mean);
    FreeTensorIfUnused(m_Variance);
    FreeTensorIfUnused(m_Gamma);
    FreeTensorIfUnused(m_Beta);
}

} //namespace armnn
