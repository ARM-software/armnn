//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonFullyConnectedFloat32Workload.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "backends/ArmComputeTensorUtils.hpp"


namespace armnn
{
using namespace armcomputetensorutils;

NeonFullyConnectedFloat32Workload::NeonFullyConnectedFloat32Workload(const FullyConnectedQueueDescriptor& descriptor,
                                                                     const WorkloadInfo& info)
    : Float32Workload<FullyConnectedQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonFullyConnectedFloat32Workload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    BuildArmComputeTensor(m_WeightsTensor, m_Data.m_Weight->GetTensorInfo());

    arm_compute::Tensor* optionalBiasTensor = nullptr;
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        BuildArmComputeTensor(m_BiasesTensor, m_Data.m_Bias->GetTensorInfo());
        optionalBiasTensor = &m_BiasesTensor;
    }

    // Construct
    m_FullyConnectedLayer.configure(
        &input, &m_WeightsTensor, optionalBiasTensor, &output, m_Data.m_Parameters.m_TransposeWeightMatrix);

    // Allocate
    InitialiseArmComputeTensorData(m_WeightsTensor, m_Data.m_Weight->GetConstTensor<float>());

    if (optionalBiasTensor)
    {
        InitialiseArmComputeTensorData(*optionalBiasTensor, m_Data.m_Bias->GetConstTensor<float>());
    }
}

void NeonFullyConnectedFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "NeonFullyConnectedFloat32Workload_Execute");
    m_FullyConnectedLayer.run();
}

} //namespace armnn


