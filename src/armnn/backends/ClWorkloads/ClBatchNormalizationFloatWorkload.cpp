//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClBatchNormalizationFloatWorkload.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "backends/ArmComputeTensorUtils.hpp"
#include "backends/ClLayerSupport.hpp"

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
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);
    const arm_compute::TensorInfo aclMeanInfo = BuildArmComputeTensorInfo(mean);
    const arm_compute::TensorInfo aclVarInfo = BuildArmComputeTensorInfo(var);
    const arm_compute::TensorInfo aclBetaInfo = BuildArmComputeTensorInfo(beta);
    const arm_compute::TensorInfo aclGammaInfo = BuildArmComputeTensorInfo(gamma);

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

    m_Layer.configure(&input,
                      &output,
                      m_Mean.get(),
                      m_Variance.get(),
                      m_Beta.get(),
                      m_Gamma.get(),
                      m_Data.m_Parameters.m_Eps);

    InitializeArmComputeClTensorDataForFloatTypes(*m_Mean, m_Data.m_Mean);
    InitializeArmComputeClTensorDataForFloatTypes(*m_Variance, m_Data.m_Variance);
    InitializeArmComputeClTensorDataForFloatTypes(*m_Beta, m_Data.m_Beta);
    InitializeArmComputeClTensorDataForFloatTypes(*m_Gamma, m_Data.m_Gamma);

    // Force Compute Library to perform the necessary copying and reshaping, after which
    // delete all the input tensors that will no longer be needed
    m_Layer.prepare();
    FreeUnusedTensors();
}

void ClBatchNormalizationFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClBatchNormalizationFloatWorkload_Execute");
    m_Layer.run();
}

void ClBatchNormalizationFloatWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_Mean);
    FreeTensorIfUnused(m_Variance);
    FreeTensorIfUnused(m_Gamma);
    FreeTensorIfUnused(m_Beta);
}

} //namespace armnn