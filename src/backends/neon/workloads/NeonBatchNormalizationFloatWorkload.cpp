//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonBatchNormalizationFloatWorkload.hpp"
#include <backendsCommon/CpuTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/ArmNN.hpp>

namespace armnn
{
using namespace armcomputetensorutils;


arm_compute::Status NeonBatchNormalizationValidate(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const TensorInfo& mean,
                                                   const TensorInfo& var,
                                                   const TensorInfo& beta,
                                                   const TensorInfo& gamma,
                                                   const BatchNormalizationDescriptor& descriptor)
{
    const DataLayout dataLayout = descriptor.m_DataLayout.GetDataLayout();

    const arm_compute::TensorInfo aclInputInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(input, dataLayout);
    const arm_compute::TensorInfo aclOutputInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(output, dataLayout);
    const arm_compute::TensorInfo aclMeanInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(mean, dataLayout);
    const arm_compute::TensorInfo aclVarInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(var, dataLayout);
    const arm_compute::TensorInfo aclBetaInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(beta, dataLayout);
    const arm_compute::TensorInfo aclGammaInfo =
          armcomputetensorutils::BuildArmComputeTensorInfo(gamma, dataLayout);

    return arm_compute::NEBatchNormalizationLayer::validate(&aclInputInfo,
                                                            &aclOutputInfo,
                                                            &aclMeanInfo,
                                                            &aclVarInfo,
                                                            &aclBetaInfo,
                                                            &aclGammaInfo,
                                                            descriptor.m_Eps);
}

NeonBatchNormalizationFloatWorkload::NeonBatchNormalizationFloatWorkload(
    const BatchNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info)
    : FloatWorkload<BatchNormalizationQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonBatchNormalizationFloatWorkload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout.GetDataLayout());
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

    m_Layer.configure(&input,
                      &output,
                      m_Mean.get(),
                      m_Variance.get(),
                      m_Beta.get(),
                      m_Gamma.get(),
                      m_Data.m_Parameters.m_Eps);

    InitializeArmComputeTensorData(*m_Mean, m_Data.m_Mean);
    InitializeArmComputeTensorData(*m_Variance, m_Data.m_Variance);
    InitializeArmComputeTensorData(*m_Gamma, m_Data.m_Gamma);
    InitializeArmComputeTensorData(*m_Beta, m_Data.m_Beta);

    // Force Compute Library to perform the necessary copying and reshaping, after which
    // delete all the input tensors that will no longer be needed
    m_Layer.prepare();
    FreeUnusedTensors();
}

void NeonBatchNormalizationFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonBatchNormalizationFloatWorkload_Execute");
    m_Layer.run();
}

void NeonBatchNormalizationFloatWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_Mean);
    FreeTensorIfUnused(m_Variance);
    FreeTensorIfUnused(m_Gamma);
    FreeTensorIfUnused(m_Beta);
}

} //namespace armnn
