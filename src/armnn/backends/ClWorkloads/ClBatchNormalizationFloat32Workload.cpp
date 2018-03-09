//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClBatchNormalizationFloat32Workload.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "backends/ArmComputeTensorUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

ClBatchNormalizationFloat32Workload::ClBatchNormalizationFloat32Workload(
    const BatchNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info)
    : Float32Workload<BatchNormalizationQueueDescriptor>(descriptor, info)
{
    BuildArmComputeTensor(m_Mean, m_Data.m_Mean->GetTensorInfo());
    BuildArmComputeTensor(m_Variance, m_Data.m_Variance->GetTensorInfo());
    BuildArmComputeTensor(m_Gamma, m_Data.m_Gamma->GetTensorInfo());
    BuildArmComputeTensor(m_Beta, m_Data.m_Beta->GetTensorInfo());

    m_Data.ValidateInputsOutputs("ClBatchNormalizationFloat32Workload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    m_Layer.configure(&input, &output, &m_Mean, &m_Variance, &m_Beta, &m_Gamma, m_Data.m_Parameters.m_Eps);

    InitialiseArmComputeClTensorData(m_Mean, m_Data.m_Mean->GetConstTensor<float>());
    InitialiseArmComputeClTensorData(m_Variance, m_Data.m_Variance->GetConstTensor<float>());
    InitialiseArmComputeClTensorData(m_Beta, m_Data.m_Beta->GetConstTensor<float>());
    InitialiseArmComputeClTensorData(m_Gamma, m_Data.m_Gamma->GetConstTensor<float>());
}

void ClBatchNormalizationFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "ClBatchNormalizationFloat32Workload_Execute");
    m_Layer.run();
}

} //namespace armnn