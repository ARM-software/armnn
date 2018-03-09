//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonBatchNormalizationFloat32Workload.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "backends/ArmComputeTensorUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

NeonBatchNormalizationFloat32Workload::NeonBatchNormalizationFloat32Workload(
    const BatchNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info)
    : Float32Workload<BatchNormalizationQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonBatchNormalizationFloat32Workload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    BuildArmComputeTensor(m_Mean, m_Data.m_Mean->GetTensorInfo());
    BuildArmComputeTensor(m_Variance, m_Data.m_Variance->GetTensorInfo());
    BuildArmComputeTensor(m_Gamma, m_Data.m_Gamma->GetTensorInfo());
    BuildArmComputeTensor(m_Beta, m_Data.m_Beta->GetTensorInfo());

    m_Layer.configure(
        &input, &output, &m_Mean, &m_Variance, &m_Beta, &m_Gamma, m_Data.m_Parameters.m_Eps);

    InitialiseArmComputeTensorData(m_Mean, m_Data.m_Mean->GetConstTensor<float>());
    InitialiseArmComputeTensorData(m_Variance, m_Data.m_Variance->GetConstTensor<float>());
    InitialiseArmComputeTensorData(m_Gamma, m_Data.m_Gamma->GetConstTensor<float>());
    InitialiseArmComputeTensorData(m_Beta, m_Data.m_Beta->GetConstTensor<float>());
}

void NeonBatchNormalizationFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "NeonBatchNormalizationFloat32Workload_Execute");
    m_Layer.run();
}

} //namespace armnn


